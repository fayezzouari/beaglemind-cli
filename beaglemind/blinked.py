#!/usr/bin/env python3

import time
import os

# Define the number of LEDs to blink
LEDs = 4

# Define the LED path
LEDPATH = '/sys/class/leds/beaglebone:green:usr'

def blink_leds():
    try:
        # Open a file for each LED
        files = []
        for i in range(LEDs):
            files.append(open(LEDPATH + str(i) + "/brightness", "w"))

        # Sequence the LEDs on and off
        while True:
            for i in range(LEDs):
                files[i].seek(0)
                files[i].write("1")  # Turn LED on
                files[i].flush()
                time.sleep(0.5)
                files[i].seek(0)
                files[i].write("0")  # Turn LED off
                files[i].flush()
                time.sleep(0.5)

    except KeyboardInterrupt:
        # Close the files when the script is interrupted
        for file in files:
            file.close()

    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred: {e}")

def main():
    try:
        # Validate the number of LEDs
        if LEDs < 1:
            print("Error: Number of LEDs must be greater than 0.")
            return

        # Validate the LED path
        if not os.path.exists(LEDPATH):
            print("Error: LED path does not exist.")
            return

        # Blink the LEDs
        blink_leds()

    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()