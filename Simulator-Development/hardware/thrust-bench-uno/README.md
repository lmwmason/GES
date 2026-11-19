# Arduino Uno thrust bench

| Pin | Function |
|---|---|
| D2 | HX711 SCK (PD_SCK) |
| D3 | HX711 DT (DOUT) |
| D4 | ESC signal for RC PWM |
| D8 | Optional PWM measurement input (ICP1) |

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

## Check PWM without an oscilloscope

Disconnect the ESC drive battery. Optional temporary jumper from the ESC
connector's white signal contact (the same electrical point as D4) to Uno D8.
Leave the common ground in place. Keep the red BEC wire isolated while the
Uno is powered by USB. Do not connect D8 to BEC power or motor wires.

Send `ARM` and then `PROBE` while the firmware reports minimum PWM, and
finish with `STOP`. The capture does not turn on PWM by itself. A normal
measurement is about `width_ns=1000000 period_ns=20000000`.
If there is no complete pulse within 150 ms, the firmware reports a timeout.
This measures edge timing at D8 on the same clock as the generator.
It does not measure signal amplitude or tell whether the ESC has armed.

## XM30FX endpoint calibration

The optional Linux console sends the same commands with single keystrokes.
Close the regular serial monitor before opening it.

```sh
uv run --no-project --with pyserial python calibrate_console.py /dev/ttyACM0
```

The console waits for the operator to press `h` and displays the connection
prompt only after the Uno acknowledges the high command. Press `l` immediately
after the two confirmation beeps, or `!` to abort.

The normal `RUN` cap remains 1500 us. `CAL HIGH` is a separate command
that can send the full 2000 us signal for at most six seconds. Use it only
with the propeller removed, motor secured, and ESC battery disconnected at
the start. Keep the Uno on USB with the red BEC power lead isolated.

1. With ESC power disconnected, send `CAL HIGH`. Connect the ESC battery
   promptly after the acknowledgement.
2. When the ESC emits its two high-endpoint confirmation beeps, immediately
   send `CAL LOW`. It outputs 1000 us for two seconds, then disables output.
   Listen for the long confirmation beep. If the expected high-endpoint
   cue does not occur, stop with `!` and disconnect the ESC battery.
3. Power-cycle the ESC with minimum PWM already present for a normal startup
   test. Confirm the long ready beep before commanding motor throttle.

If `CAL LOW` is not sent during the high window, the firmware turns D4 off.
`STOP` and `!` also turn D4 off during calibration.
[XM30FX manual](https://m.xcopter.com/product/pt/xm30fx/xm30fx.pdf)
