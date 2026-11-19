#![no_std]

pub const IDLE_US: u16 = 1000;
pub const MAX_US: u16 = 1500;
pub const FRAME_MS: u32 = 20;
const ARM_FRAMES: u16 = 150;
const READY_FRAMES: u16 = 500;
const STOP_FRAMES: u16 = 50;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Esc {
    Off,
    Arming(u16),
    Ready(u16),
    Running(u16, u16),
    Stopping(u16),
}

impl Esc {
    pub fn arm(&mut self) -> bool {
        if *self != Self::Off {
            return false;
        }
        *self = Self::Arming(ARM_FRAMES);
        true
    }

    pub fn run(&mut self, pulse_us: u16, duration_ms: u16) -> bool {
        if !matches!(self, Self::Ready(_))
            || !(IDLE_US + 1..=MAX_US).contains(&pulse_us)
            || !(100..=10_000).contains(&duration_ms)
        {
            return false;
        }

        *self = Self::Running(pulse_us, duration_ms.div_ceil(FRAME_MS as u16));
        true
    }

    pub fn stop(&mut self) {
        if !matches!(*self, Self::Off | Self::Stopping(_)) {
            *self = Self::Stopping(STOP_FRAMES);
        }
    }

    pub fn next_frame(&mut self) -> u16 {
        let (pulse, next) = match *self {
            Self::Off => (0, Self::Off),
            Self::Arming(1) => (IDLE_US, Self::Ready(READY_FRAMES)),
            Self::Arming(frames) => (IDLE_US, Self::Arming(frames - 1)),
            Self::Ready(1) | Self::Stopping(1) => (IDLE_US, Self::Off),
            Self::Ready(frames) => (IDLE_US, Self::Ready(frames - 1)),
            Self::Running(pulse, 1) => (pulse, Self::Ready(READY_FRAMES)),
            Self::Running(pulse, frames) => (pulse, Self::Running(pulse, frames - 1)),
            Self::Stopping(frames) => (IDLE_US, Self::Stopping(frames - 1)),
        };
        *self = next;
        pulse
    }
}

pub fn sign_extend_hx711(raw: u32) -> i32 {
    ((raw << 8) as i32) >> 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_bounded_run_can_raise_throttle() {
        let mut esc = Esc::Off;
        assert_eq!(esc.next_frame(), 0);
        assert!(!esc.run(1200, 1000));
        assert!(esc.arm());
        assert!(!esc.run(1200, 1000));
        for _ in 0..ARM_FRAMES {
            assert_eq!(esc.next_frame(), IDLE_US);
        }
        assert!(!esc.run(1501, 1000));
        assert!(!esc.run(1200, 10001));
        assert!(esc.run(1200, 105));
        for _ in 0..6 {
            assert_eq!(esc.next_frame(), 1200);
        }
        assert_eq!(esc.next_frame(), IDLE_US);
    }

    #[test]
    fn stop_overrides_running_and_disables_signal_after_idle_frames() {
        let mut esc = Esc::Ready(1);
        assert!(esc.run(1200, 2000));
        esc.stop();
        assert!(!esc.run(1300, 1000));
        for _ in 0..STOP_FRAMES {
            assert_eq!(esc.next_frame(), IDLE_US);
        }
        assert_eq!(esc.next_frame(), 0);
    }

    #[test]
    fn hx711_is_signed_24_bit() {
        assert_eq!(sign_extend_hx711(0x7fffff), 8388607);
        assert_eq!(sign_extend_hx711(0x800000), -8388608);
        assert_eq!(sign_extend_hx711(0xffffff), -1);
    }
}
