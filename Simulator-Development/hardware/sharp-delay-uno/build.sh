#!/bin/sh
set -eu
cd "$(dirname "$0")"

if ! command -v avr-gcc >/dev/null 2>&1 || ! command -v avr-objcopy >/dev/null 2>&1; then
    data_dir=$(arduino-cli config get directories.data)
    tool_dir=$(find "$data_dir/packages/arduino/tools/avr-gcc" -mindepth 3 -maxdepth 3 \
        -type f -name avr-gcc -print | sort -V | tail -n 1)
    if [ -z "$tool_dir" ]; then
        echo "Install the Arduino AVR core or make avr-gcc and avr-objcopy available." >&2
        exit 1
    fi
    export PATH="$(dirname "$tool_dir"):$PATH"
fi

cargo build --locked --release
output=target/avr-none/release/sharp-delay-uno
avr-objcopy -O ihex -R .eeprom "$output.elf" "$output.hex"
printf '%s\n' "$output.hex"

if [ "$#" -eq 1 ]; then
    arduino-cli upload --fqbn arduino:avr:uno --port "$1" --input-file "$output.hex" --verify
elif [ "$#" -gt 1 ]; then
    echo "Usage: sh build.sh [serial-port]" >&2
    exit 2
fi
