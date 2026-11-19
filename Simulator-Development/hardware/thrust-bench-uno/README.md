# Arduino Uno thrust bench

| Pin | Function |
|---|---|
| D2 | HX711 SCK (PD_SCK) |
| D3 | HX711 DT (DOUT) |
| D4 | ESC signal for RC PWM |

This firmware targets the 16 MHz ATmega328P Uno, HX711 channel A at gain 128,
and a unidirectional ESC that accepts 50 Hz RC PWM with 1000 us as its
documented stop command. Check the ESC manual and signal voltage compatibility
before enabling its motor supply. The ESC drive power comes from its rated
supply, and the ESC signal ground and Uno ground must be common. Use a guarded
fixture and an independent physical motor cutoff. Test commands and pulse
behavior without a propeller first.

[HX711 datasheet](https://cdn.sparkfun.com/assets/b/f/5/a/e/hx711F_EN.pdf) ·
[RC PWM protocol](https://docs.px4.io/main/en/esc/esc_protocols)

## Build and upload

```sh
cd Simulator-Development/hardware/thrust-bench-uno
sh build.sh                  # Compile ELF and HEX
sh build.sh /dev/ttyACM0     # Compile, upload, and verify
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

## Record a run

Open the serial monitor before testing. Opening it can reset the Uno and
clears the tare. Send uppercase commands, followed by Enter.

- `STATUS` reports the output state, command pulse width, and HX711 freshness.
- `TARE` averages the next 16 valid samples with PWM off. Tare with the motor
  at rest and the fixture fully assembled. Wait for the result before arming.
- `ARM` starts a 3-second interval at the configured 1000 us minimum. Wait for
  `STATUS` to report `ready`, then send the run within 10 seconds.
- `RUN 1100 3000` requests a 1100 us pulse for 3000 ms. Runs are capped at
  1500 us and 10 seconds. On expiry the controller returns to minimum, then
  disables PWM after 10 seconds of inactivity.
- `STOP` or a single `!` interrupts a run. The firmware drives the signal low
  immediately, sends the configured minimum pulses for about one second, then
  disables PWM. Loss of HX711 samples for 500 ms or a saturated conversion
  also requests stop. A serial disconnect does not extend an in-progress run.

On boot D4 is low. The load cell samples stream immediately as
`time_ms,pwm_us,raw_counts,net_counts`. Timestamp is when the firmware detects
data-ready, and `pwm_us` is the software's current frame command. It is not an
RPM reading. Lines beginning with `#` describe checks or commands. Have a known
mass and the measured calibration slope before converting counts to newtons.
Record height and independently measured RPM with each steady segment.
