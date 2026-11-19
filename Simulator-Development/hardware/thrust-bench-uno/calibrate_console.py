#!/usr/bin/env python3
"""Single-key local console for XM30FX calibration on the thrust-bench Uno."""

import argparse
import os
import select
import sys
import termios
import time
import tty

import serial


def send(port, command):
    port.write(command + b"\n")
    port.flush()


def verify_idle(port):
    time.sleep(2.5)
    port.reset_input_buffer()
    send(port, b"STATUS")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        line = port.readline().strip()
        if line.startswith(b"# esc="):
            if line != b"# esc=off pwm_us=0 hx711_fresh=true":
                raise RuntimeError(line.decode(errors="replace"))
            return
    raise RuntimeError("Did not receive Uno status. Close any other serial monitor.")


def console(port):
    verify_idle(port)
    print(
        "Propeller removed, motor secured, ESC battery UNPLUGGED, Uno on USB.\n"
        "h  Start the 6 s high-endpoint window. Plug in battery after 'CONNECT NOW'.\n"
        "l  Send low as soon as the ESC makes two high-endpoint confirmation beeps.\n"
        "!  Abort immediately. q  Stop and exit.\n"
        "If the expected cue is absent, abort and unplug the battery.",
        flush=True,
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    high_started = None
    timed_out = False
    waiting_for_low_reply = False
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([fd, port.fileno()], [], [], 0.1)
            if fd in readable:
                key = os.read(fd, 1).lower()
                if key in (b"q", b""):
                    break
                if key == b"h":
                    send(port, b"CAL HIGH")
                elif key == b"l":
                    if high_started is not None:
                        elapsed = time.monotonic() - high_started
                        print(f"LOW key at {elapsed:.2f} s after HIGH acknowledgement.", flush=True)
                    waiting_for_low_reply = True
                    send(port, b"CAL LOW")
                elif key == b"s":
                    send(port, b"STATUS")
                elif key == b"!":
                    send(port, b"!")
                    high_started = None
            if port.fileno() in readable:
                line = port.readline().strip()
                if line.startswith(b"#"):
                    print(line.decode(errors="replace"), flush=True)
                    if line.startswith(b"# CAL HIGH:"):
                        high_started = time.monotonic()
                        timed_out = False
                        waiting_for_low_reply = False
                        print("CONNECT NOW. Press l immediately after the two beeps.", flush=True)
                    elif line.startswith(b"# CAL LOW:"):
                        high_started = None
                        waiting_for_low_reply = False
                        print("Listen for the long low-endpoint confirmation beep.", flush=True)
                    elif line.startswith(b"# rejected") and waiting_for_low_reply:
                        waiting_for_low_reply = False
                        send(port, b"STATUS")
            if high_started is not None and not timed_out and time.monotonic() - high_started >= 6:
                timed_out = True
                print("High window ended. If not acknowledged, unplug the ESC battery.", flush=True)
                send(port, b"STATUS")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        send(port, b"STOP")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="/dev/ttyACM0")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Use an interactive terminal")
    with serial.Serial(args.port, 115200, timeout=0.1, write_timeout=1) as port:
        console(port)


if __name__ == "__main__":
    main()
