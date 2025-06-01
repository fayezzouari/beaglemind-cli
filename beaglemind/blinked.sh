#!/bin/bash

# Set error handling to exit on error
set -e

# Define the LED path and the default LED number
LED_PATH='/sys/class/leds/beaglebone:green:usr'
LED_NUMBER=${1:-1}

# Check if the LED number is valid
if ! [[ $LED_NUMBER =~ ^[0-9]+$ ]] || (( LED_NUMBER < 0 || LED_NUMBER > 3 )); then
    echo "Error: Invalid LED number. Please enter a number between 0 and 3."
    exit 1
fi

# Check if the LED path exists
if [ ! -d "$LED_PATH" ]; then
    echo "Error: LED path does not exist."
    exit 1
fi

# Define the brightness levels
BRIGHTNESS_ON="1"
BRIGHTNESS_OFF="0"

# Usage information
echo "Usage: $0 <LED number>"

# Blink the LED
while true; do
    echo "$BRIGHTNESS_ON" > "${LED_PATH}${LED_NUMBER}/brightness"
    sleep 0.5
    echo "$BRIGHTNESS_OFF" > "${LED_PATH}${LED_NUMBER}/brightness"
    sleep 0.5
done