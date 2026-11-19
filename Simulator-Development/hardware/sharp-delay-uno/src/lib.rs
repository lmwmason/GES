#![no_std]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thresholds {
    pub rise: u16,
    pub fall: u16,
}

impl Thresholds {
    pub fn from_calibration(far: u16, near: u16) -> Option<Self> {
        let span = near.checked_sub(far)?;
        if span < 20 {
            return None;
        }

        let midpoint = far + span / 2;
        let half_band = span / 10;
        Some(Self {
            rise: midpoint + half_band,
            fall: midpoint - half_band,
        })
    }

    pub fn next_state(self, near: bool, adc: u16) -> bool {
        if near {
            adc > self.fall
        } else {
            adc >= self.rise
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_preserves_state_inside_band() {
        let thresholds = Thresholds::from_calibration(120, 220).unwrap();
        assert_eq!(
            thresholds,
            Thresholds {
                rise: 180,
                fall: 160
            }
        );
        assert!(!thresholds.next_state(false, 170));
        assert!(thresholds.next_state(true, 170));
        assert!(thresholds.next_state(false, 180));
        assert!(!thresholds.next_state(true, 160));
    }

    #[test]
    fn reversed_or_too_close_calibration_is_rejected() {
        assert!(Thresholds::from_calibration(210, 180).is_none());
        assert!(Thresholds::from_calibration(180, 199).is_none());
    }
}
