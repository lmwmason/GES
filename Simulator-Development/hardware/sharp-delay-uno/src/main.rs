#![no_std]
#![no_main]

use arduino_hal::prelude::*;
use panic_halt as _;
use sharp_delay_uno::Thresholds;

const US_PER_TICK: u32 = 4;
const SAMPLE_TICKS: u32 = 250;
const CALIBRATION_SAMPLES: u16 = 128;

struct Clock {
    timer: arduino_hal::pac::TC1,
    last: u16,
    ticks: u32,
}

impl Clock {
    fn new(timer: arduino_hal::pac::TC1) -> Self {
        // Uno's 16 MHz clock divided by 64 gives a Timer1 tick of 4 us.
        timer.tccr1b().write(|w| w.cs1().prescale_64());
        let last = timer.tcnt1().read().bits();
        Self {
            timer,
            last,
            ticks: 0,
        }
    }

    fn now(&mut self) -> u32 {
        let count = self.timer.tcnt1().read().bits();
        self.ticks = self
            .ticks
            .wrapping_add(u32::from(count.wrapping_sub(self.last)));
        self.last = count;
        self.ticks
    }
}

#[arduino_hal::entry]
fn main() -> ! {
    let peripherals = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(peripherals);
    let mut serial = arduino_hal::default_serial!(peripherals, pins, 115200);
    let mut led = pins.d2.into_output();
    led.set_low();

    let mut adc = arduino_hal::Adc::new(peripherals.ADC, Default::default());
    let a0 = pins.a0.into_analog_input(&mut adc);
    let mut clock = Clock::new(peripherals.TC1);
    for _ in 0..10 {
        arduino_hal::delay_ms(50);
        clock.now();
    }

    ufmt::uwriteln!(&mut serial, "# sharp-delay-uno 0.1.0; A0 sensor, D2 LED").unwrap_infallible();
    ufmt::uwriteln!(
        &mut serial,
        "# f=calibrate far, n=calibrate near, r=run, x=stop, a=read A0"
    )
    .unwrap_infallible();

    let mut far = None;
    let mut near = None;
    let mut active: Option<Thresholds> = None;
    let mut led_near = false;
    let mut last_sample = clock.now();

    loop {
        let now = clock.now();
        if let Some(thresholds) = active {
            if now.wrapping_sub(last_sample) >= SAMPLE_TICKS {
                last_sample = now;
                let sample = a0.analog_read(&mut adc);
                let next_near = thresholds.next_state(led_near, sample);
                if next_near != led_near {
                    led_near = next_near;
                    if led_near {
                        led.set_high();
                    } else {
                        led.set_low();
                    }
                    let timestamp_us = clock.now().wrapping_mul(US_PER_TICK);
                    let edge = if led_near { "rise" } else { "fall" };
                    ufmt::uwriteln!(&mut serial, "{},{},{}", timestamp_us, edge, sample)
                        .unwrap_infallible();
                }
            }
        }

        match serial.read() {
            Ok(command @ (b'f' | b'n')) => {
                active = None;
                led.set_low();
                let mut sum = 0u32;
                for _ in 0..CALIBRATION_SAMPLES {
                    sum += u32::from(a0.analog_read(&mut adc));
                    arduino_hal::delay_ms(2);
                    clock.now();
                }
                let mean = (sum / u32::from(CALIBRATION_SAMPLES)) as u16;
                if command == b'f' {
                    far = Some(mean);
                    ufmt::uwriteln!(&mut serial, "# far_adc={}", mean).unwrap_infallible();
                } else {
                    near = Some(mean);
                    ufmt::uwriteln!(&mut serial, "# near_adc={}", mean).unwrap_infallible();
                }
            }
            Ok(b'r') => {
                let thresholds = far
                    .zip(near)
                    .and_then(|(f, n)| Thresholds::from_calibration(f, n));
                if let Some(thresholds) = thresholds {
                    let initial = a0.analog_read(&mut adc);
                    if initial > thresholds.fall {
                        ufmt::uwriteln!(&mut serial, "# place far target before run")
                            .unwrap_infallible();
                        continue;
                    }
                    led_near = false;
                    led.set_low();
                    ufmt::uwriteln!(
                        &mut serial,
                        "# armed rise_adc={} fall_adc={}",
                        thresholds.rise,
                        thresholds.fall
                    )
                    .unwrap_infallible();
                    ufmt::uwriteln!(&mut serial, "led_time_us,direction,adc").unwrap_infallible();
                    last_sample = clock.now();
                    active = Some(thresholds);
                } else {
                    ufmt::uwriteln!(
                        &mut serial,
                        "# calibrate f and n; near must exceed far by at least 20 ADC counts"
                    )
                    .unwrap_infallible();
                }
            }
            Ok(b'x') => {
                active = None;
                led.set_low();
                ufmt::uwriteln!(&mut serial, "# stopped").unwrap_infallible();
            }
            Ok(b'a') if active.is_none() => {
                ufmt::uwriteln!(&mut serial, "# adc={}", a0.analog_read(&mut adc))
                    .unwrap_infallible();
            }
            _ => {}
        }
    }
}
