# Wallpaper Clock

A Python script for a live-updating astronomical clock as a wallpaper.

Tested on Ubuntu with Python 3.11.

## How to use

#### To run once:

1. Change `LATLONG` (line 94) to your latitude and longitude.
   * Unfortunately, the projection requires that the math be done differently in the northern and southern hemispheres. As this is mostly for personal use, the script currently only works for the northern hemisphere.
3. Run `nohup timeand.py &` in this directory. This will generate `test.png`.
4. Set `test.png` as your wallpaper. The program will keep running and update this image every 2 minutes.

#### To run every time the computer starts up:

1. In `startclock.sh`, set `# path to this directory` to the path to this directory.
2. Add `startclock.sh` as a startup application.

## How to read

For stylistic reasons, the clock contains no text.

* **Gray ellipse:** The horizon at the current time.
* **Large white dot / Unfilled gray dot:** The position of the sun and moon in the sky.
* **Innermost ticks:** The time of day. The longest tick (usually close to the sun) marks midnight. The gray line attached to the north of the horizon ellipse points to the current time.
* **Inner ring:** The 24 solar terms. The 4 thickest lines mark the winter solstice(top), spring equinox(right), summer solstice(bottom), and autumn equinox(left).
* **Outer ring:** Dates of the year. The thickest line near the top marks the start of the year. Each thick line marks the start of the first day of each month. The white line attached to the sun points to the current date.
* **Two circles to the left:** Positions of each planet on the ecliptic plane, as opposed to in the sky.

## Example Image

10:00AM at June 28th, 2024

![test000](https://github.com/yoonshik1205/wallpaper_clock/assets/30615279/04afddc4-5356-4d19-b2e8-1d8268a4e8f4)


