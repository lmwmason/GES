#!/usr/bin/env python3
"""Single-key local console for XM30FX calibration on the thrust-bench Uno."""

import argparse
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
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([fd, port.fileno()], [], [])
            if fd in readable:
                key = sys.stdin.read(1).lower()
                if key in ("q", ""):
                    break
                if key == "h":
                    send(port, b"CAL HIGH")
                elif key == "l":
                    send(port, b"CAL LOW")
                elif key == "!":
                    send(port, b"!")
            if port.fileno() in readable:
                line = port.readline().strip()
                if line.startswith(b"#"):
                    print(line.decode(errors="replace"), flush=True)
                    if line.startswith(b"# CAL HIGH:"):
                        print("CONNECT NOW. Press l immediately after the two beeps.", flush=True)
                    elif line.startswith(b"# CAL LOW:"):
                        print("Listen for the long low-endpoint confirmation beep.", flush=True)
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
