#![no_std]
#![no_main]
#![feature(abi_avr_interrupt)]

use arduino_hal::port::{Pin, mode::Output};
use arduino_hal::prelude::*;
use avr_device::interrupt::{self, Mutex};
use core::cell::RefCell;
use panic_halt as _;
use thrust_bench_uno::{Esc, FRAME_MS, IDLE_US, sign_extend_hx711};

const FRAME_TICKS: u16 = (FRAME_MS * 2000) as u16;
const RATE_TIMEOUT_MS: u32 = 500;
const TARE_SAMPLES: u8 = 16;
const CAPTURE_TIMEOUT_MS: u32 = 150;

#[derive(Default)]
struct Capture {
    started_ms: Option<u32>,
    last_rise: Option<u16>,
    period: Option<u16>,
    width: Option<u16>,
    waiting_for_fall: bool,
}

struct Pwm {
    esc: Esc,
    timer: arduino_hal::pac::TC1,
    pin: Pin<Output>,
    high: bool,
    frame_ms: u32,
    frame_start: u16,
    applied_us: u16,
    capture: Capture,
}

static PWM: Mutex<RefCell<Option<Pwm>>> = Mutex::new(RefCell::new(None));

fn with_pwm<T>(f: impl FnOnce(&mut Pwm) -> T) -> T {
    interrupt::free(|cs| f(PWM.borrow(cs).borrow_mut().as_mut().unwrap()))
}

impl Pwm {
    fn now_ms(&self) -> u32 {
        let timer = self.timer.tcnt1().read().bits();
        self.frame_ms
            .wrapping_add(u32::from(timer.wrapping_sub(self.frame_start)) / 2000)
    }

    fn schedule(&self, tick: u16) {
        self.timer.ocr1a().write(|w| w.set(tick));
    }

    fn stop(&mut self) {
        self.pin.set_low();
        self.applied_us = 0;
        self.esc.stop();
        // Leave high set until the existing falling-edge timer event. It
        // will schedule the normal frame boundary, without raising the pin.
    }

    fn start_capture(&mut self) {
        self.capture = Capture {
            started_ms: Some(self.now_ms()),
            ..Capture::default()
        };
        self.timer.tccr1b().modify(|_, w| w.ices1().set_bit());
        self.timer.tifr1().write(|w| w.icf1().set_bit());
        self.timer.timsk1().modify(|_, w| w.icie1().set_bit());
    }

    fn stop_capture(&self) {
        self.timer.timsk1().modify(|_, w| w.icie1().clear_bit());
    }
}

#[avr_device::interrupt(atmega328p)]
fn TIMER1_CAPT() {
    with_pwm(|pwm| {
        let tick = pwm.timer.icr1().read().bits();
        let capture = &mut pwm.capture;
        if capture.waiting_for_fall {
            if let Some(rise) = capture.last_rise {
                if capture.period.is_some() {
                    capture.width = Some(tick.wrapping_sub(rise));
                    pwm.stop_capture();
                    return;
                }
            }
            pwm.timer.tccr1b().modify(|_, w| w.ices1().set_bit());
        } else {
            capture.period = capture.last_rise.map(|rise| tick.wrapping_sub(rise));
            capture.last_rise = Some(tick);
            pwm.timer.tccr1b().modify(|_, w| w.ices1().clear_bit());
        }
        capture.waiting_for_fall = !capture.waiting_for_fall;
        pwm.timer.tifr1().write(|w| w.icf1().set_bit());
    });
}

#[avr_device::interrupt(atmega328p)]
fn TIMER1_COMPA() {
    with_pwm(|pwm| {
        if pwm.high {
            pwm.pin.set_low();
            pwm.high = false;
            pwm.schedule(pwm.frame_start.wrapping_add(FRAME_TICKS));
        } else {
            pwm.frame_start = pwm.timer.tcnt1().read().bits();
            pwm.frame_ms = pwm.frame_ms.wrapping_add(FRAME_MS);
            pwm.applied_us = pwm.esc.next_frame();
            if pwm.applied_us == 0 {
                pwm.schedule(pwm.frame_start.wrapping_add(FRAME_TICKS));
            } else {
                pwm.pin.set_high();
                pwm.high = true;
                pwm.schedule(pwm.frame_start.wrapping_add(pwm.applied_us * 2));
            }
        }
    });
}

enum Command {
    Arm,
    Run(u16, u16),
    Stop,
    Tare,
    Status,
    Probe,
}

fn parse(line: &str) -> Option<Command> {
    let mut words = line.split_ascii_whitespace();
    let command = match words.next()? {
        "ARM" => Command::Arm,
        "RUN" => Command::Run(words.next()?.parse().ok()?, words.next()?.parse().ok()?),
        "STOP" => Command::Stop,
        "TARE" => Command::Tare,
        "STATUS" => Command::Status,
        "PROBE" => Command::Probe,
        _ => return None,
    };
    words.next().is_none().then_some(command)
}

#[arduino_hal::entry]
fn main() -> ! {
    let peripherals = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(peripherals);
    let mut serial = arduino_hal::default_serial!(peripherals, pins, 115200);
    let mut sck = pins.d2.into_output();
    sck.set_low();
    let data = pins.d3.into_floating_input();
    let mut esc_pin = pins.d4.into_output();
    esc_pin.set_low();
    let _capture_pin = pins.d8.into_floating_input();

    let timer = peripherals.TC1;
    timer.tccr1b().write(|w| w.cs1().prescale_8());
    let start = timer.tcnt1().read().bits();
    timer
        .ocr1a()
        .write(|w| w.set(start.wrapping_add(FRAME_TICKS)));
    timer.tifr1().write(|w| w.ocf1a().set_bit());
    timer.timsk1().write(|w| w.ocie1a().set_bit());
    interrupt::free(|cs| {
        *PWM.borrow(cs).borrow_mut() = Some(Pwm {
            esc: Esc::Off,
            pin: esc_pin.downgrade(),
            timer,
            high: false,
            frame_ms: 0,
            frame_start: start,
            applied_us: 0,
            capture: Capture::default(),
        });
    });
    // Timer state and output pin are initialized before enabling interrupts.
    unsafe { interrupt::enable() };

    ufmt::uwriteln!(
        &mut serial,
        "# thrust-bench-uno; HX711 D2/D3; ESC PWM D4 off"
    )
    .unwrap_infallible();
    ufmt::uwriteln!(&mut serial, "time_ms,pwm_us,raw_counts,net_counts").unwrap_infallible();

    let mut last_sample = None;
    let mut tare = 0i32;
    let mut tare_remaining = 0;
    let mut tare_sum = 0i32;
    let mut line = [0u8; 32];
    let mut used = 0usize;
    let mut overflow = false;
    // A conversion may already be waiting when the Uno starts.
    let mut dt_was_high = true;

    loop {
        let now = with_pwm(|pwm| pwm.now_ms());
        let capture_result = with_pwm(|pwm| {
            let start = pwm.capture.started_ms?;
            if let (Some(width), Some(period)) = (pwm.capture.width, pwm.capture.period) {
                pwm.capture.started_ms = None;
                return Some(Some((width, period)));
            }
            if now.wrapping_sub(start) >= CAPTURE_TIMEOUT_MS {
                pwm.stop_capture();
                pwm.capture.started_ms = None;
                return Some(None);
            }
            None
        });
        match capture_result {
            Some(Some((width, period))) => {
                // Timer1 runs at 2 MHz.
                ufmt::uwriteln!(
                    &mut serial,
                    "# probe width_ns={} period_ns={}",
                    u32::from(width) * 500,
                    u32::from(period) * 500
                )
                .unwrap_infallible();
            }
            Some(None) => {
                ufmt::uwriteln!(&mut serial, "# probe: no complete PWM on D8").unwrap_infallible();
            }
            None => {}
        }
        let fresh = last_sample.is_some_and(|time| now.wrapping_sub(time) <= RATE_TIMEOUT_MS);
        if !fresh {
            let stopped = with_pwm(|pwm| match pwm.esc {
                Esc::Off | Esc::Stopping(_) => false,
                _ => {
                    pwm.stop();
                    true
                }
            });
            if stopped {
                ufmt::uwriteln!(&mut serial, "# stopped: no recent HX711 sample")
                    .unwrap_infallible();
            }
        }

        // Require a ready transition. A permanently low/disconnected data
        // line cannot be mistaken for repeated fresh conversions.
        if data.is_high() {
            dt_was_high = true;
        } else if dt_was_high {
            let (sample_time, pulse_us) = with_pwm(|pwm| (pwm.now_ms(), pwm.applied_us));
            let mut raw = 0u32;
            for _ in 0..24 {
                let bit = interrupt::free(|_| {
                    sck.set_high();
                    arduino_hal::delay_us(1);
                    let bit = data.is_high();
                    sck.set_low();
                    bit
                });
                raw = (raw << 1) | u32::from(bit);
                arduino_hal::delay_us(1);
            }
            interrupt::free(|_| {
                // Pulse 25 selects channel A at gain 128.
                sck.set_high();
                arduino_hal::delay_us(1);
                sck.set_low();
            });
            arduino_hal::delay_us(1);

            let counts = sign_extend_hx711(raw);
            dt_was_high = data.is_high();
            if !dt_was_high || counts == -8388608 || counts == 8388607 {
                last_sample = None;
                tare_remaining = 0;
                with_pwm(|pwm| pwm.stop());
                ufmt::uwriteln!(
                    &mut serial,
                    "# stopped: HX711 saturation or invalid conversion"
                )
                .unwrap_infallible();
            } else {
                last_sample = Some(sample_time);
                if tare_remaining > 0 {
                    tare_sum += counts;
                    tare_remaining -= 1;
                    if tare_remaining == 0 {
                        tare = tare_sum / i32::from(TARE_SAMPLES);
                        ufmt::uwriteln!(&mut serial, "# tare={}", tare).unwrap_infallible();
                    }
                }
                ufmt::uwriteln!(
                    &mut serial,
                    "{},{},{},{}",
                    sample_time,
                    pulse_us,
                    counts,
                    counts - tare
                )
                .unwrap_infallible();
            }
        }

        let Ok(byte) = serial.read() else { continue };
        if byte == b'!' {
            with_pwm(|pwm| pwm.stop());
            used = 0;
            overflow = false;
            ufmt::uwriteln!(&mut serial, "# stop requested").unwrap_infallible();
            continue;
        }
        if byte != b'\n' && byte != b'\r' {
            if used < line.len() {
                line[used] = byte;
                used += 1;
            } else {
                overflow = true;
            }
            continue;
        }
        if used == 0 {
            continue;
        }

        let command = if overflow {
            None
        } else {
            core::str::from_utf8(&line[..used]).ok().and_then(parse)
        };
        used = 0;
        overflow = false;
        match command {
            Some(Command::Arm) if fresh && tare_remaining == 0 && with_pwm(|pwm| pwm.esc.arm()) => {
                ufmt::uwriteln!(&mut serial, "# arming: 3 s at {} us; use STATUS", IDLE_US)
                    .unwrap_infallible();
            }
            Some(Command::Run(pulse, duration))
                if fresh && with_pwm(|pwm| pwm.esc.run(pulse, duration)) =>
            {
                ufmt::uwriteln!(
                    &mut serial,
                    "# run: {} us for {} ms (rounded up to frame)",
                    pulse,
                    duration
                )
                .unwrap_infallible();
            }
            Some(Command::Stop) => {
                with_pwm(|pwm| pwm.stop());
                ufmt::uwriteln!(&mut serial, "# stop requested").unwrap_infallible();
            }
            Some(Command::Tare) if fresh && with_pwm(|pwm| pwm.esc == Esc::Off) => {
                tare_sum = 0;
                tare_remaining = TARE_SAMPLES;
                ufmt::uwriteln!(&mut serial, "# tare started").unwrap_infallible();
            }
            Some(Command::Status) => {
                let (state, pulse) = with_pwm(|pwm| (pwm.esc, pwm.applied_us));
                let name = match state {
                    Esc::Off => "off",
                    Esc::Arming(_) => "arming",
                    Esc::Ready(_) => "ready",
                    Esc::Running(_, _) => "running",
                    Esc::Stopping(_) => "stopping",
                };
                ufmt::uwriteln!(
                    &mut serial,
                    "# esc={} pwm_us={} hx711_fresh={}",
                    name,
                    pulse,
                    fresh
                )
                .unwrap_infallible();
            }
            Some(Command::Probe) => with_pwm(|pwm| {
                if !matches!(pwm.esc, Esc::Running(..)) {
                    pwm.start_capture();
                }
            }),
            _ => {
                ufmt::uwriteln!(
                    &mut serial,
                    "# rejected; STATUS / PROBE / ARM / RUN <1001..1500> <100..10000ms> / TARE / STOP"
                )
                .unwrap_infallible();
            }
        }
    }
}
