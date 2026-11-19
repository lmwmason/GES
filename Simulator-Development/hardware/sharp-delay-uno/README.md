# Sharp sensor delay measurement on Arduino Uno

The target is a 16 MHz Arduino Uno with ATmega328P. The Sharp GP2Y0A41SK0F
has an analog infrared triangulation output and a rated range of 4–30 cm.
Its internal measurement cycle does not set the delay of any single change
experiment, because a distance transition can occur anywhere in that cycle.
[Sharp datasheet](https://www.sharp.co.jp/products/device/doc/opto/gp2y0a41sk_e.pdf)

## Wiring

- Sharp Vout to A0. Supply and ground must match the sensor's datasheet.
  Use a common ground and the recommended supply decoupling.
- Active-high LED on D2 with a series resistor, connected to ground.
- The camera sees the physical target transition and the LED in the same view.
  Keep the target heights within the rated range and avoid mixed returns from
  other objects behind the target.

## Build and upload

Install Rust via rustup and make `avr-gcc` and `avr-objcopy` available,
or install the Arduino AVR core with `arduino-cli core install arduino:avr`.
The build script can discover the AVR tools installed by Arduino CLI.

```sh
cd Simulator-Development/hardware/sharp-delay-uno
sh build.sh                   # Build ELF and HEX only
sh build.sh /dev/ttyACM0       # Build, upload, and verify
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

## Measurement

1. Place the far target at 25 cm and send lowercase `f` over serial.
2. Place the near target at 15 cm and send `n`. Hold each target still
   while its approximately 0.27-second calibration completes.
3. Restore the far target and send `r`. Wait for the armed message.
4. Alternate the targets, pausing at each endpoint. D2 turns on for a nearer
   transition and off for a farther transition. Stop with `x`.
   While stopped, `a` reports the live ADC value.
5. In each trial count the frames from the visible physical transition to
   the LED edge. At 1000 fps, one frame interval is 1 ms. Record both
   directions, the threshold values, target geometry, and camera settings.

Calibration uses the mean of 128 endpoint samples. Arming computes a midpoint
and directional hysteresis thresholds from those two endpoint values. After
arming, the firmware samples A0 at approximately 1 kHz. It applies the
threshold to each individual sample and toggles D2 before printing the serial
event. No moving average is applied to the samples in this path. Do not send
serial commands during a trial. Open the monitor before calibration because
opening a new serial session can reset the Uno.

The serial event stream contains `led_time_us,direction,adc`. The timestamp
is the MCU clock at the LED edge and helps identify trials. Without a physical
reference event on the MCU clock, this timestamp alone does not give latency.
Use the camera for the physical transition reference. The result includes the
sensor, ADC sampling, and threshold decision under this setup.

For independent host tests of the threshold logic:

```sh
rustc +stable --test src/lib.rs -o /tmp/ges-sharp-threshold-tests
/tmp/ges-sharp-threshold-tests
```
