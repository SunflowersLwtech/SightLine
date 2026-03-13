import os
import re

import cairosvg
import imageio.v2 as imageio

# Configuration
INPUT_SVG = 'assets/system-architecture.svg'
OUTPUT_GIF = 'assets/system-architecture.gif'
FRAMES = 12
DURATION = 0.1  # seconds per frame

def create_gif():
    print(f"Reading {INPUT_SVG}...")
    with open(INPUT_SVG, 'r') as f:
        svg_content = f.read()

    # FIX: Remove missing marker references that cause cairosvg to crash
    # The SVG references #arrowGreen but doesn't define it in <defs>
    svg_content = re.sub(r'marker-end="url\(#[^\)]+\)"', '', svg_content)

    images = []

    # We will animate the 'stroke-dashoffset' to create a flowing effect on dashed lines
    # The patterns are roughly length 6-8. A 24-unit shift covers multiples of 3, 4, 6, 8 nicely.
    # Let's use 24 frames shifting by -1 each time, or 12 frames shifting by -2.

    print("Generating frames...")
    for i in range(FRAMES):
        # Calculate offset. We shift backwards to make it look like it's flowing forward usually.
        offset = i * -2

        # Inject stroke-dashoffset into elements that have stroke-dasharray
        # We use a regex substitution callback to append the offset
        def add_offset(match):
            full_match = match.group(0)
            # If dashoffset already exists, replace it; otherwise append it
            if 'stroke-dashoffset' in full_match:
                return re.sub(r'stroke-dashoffset="[^"]*"', f'stroke-dashoffset="{offset}"', full_match)
            else:
                return f'{full_match} stroke-dashoffset="{offset}"'

        # Target lines with dasharray
        frame_svg = re.sub(r'stroke-dasharray="[^"]*"', add_offset, svg_content)

        try:
            # Convert to PNG in memory
            png_data = cairosvg.svg2png(bytestring=frame_svg.encode('utf-8'))

            # Append to image list
            images.append(imageio.imread(png_data))
            print(f"  - Frame {i+1}/{FRAMES} rendered")
        except Exception as e:
            print(f"Error rendering frame {i}: {e}")
            break

    if images:
        print(f"Saving GIF to {OUTPUT_GIF}...")
        imageio.mimsave(OUTPUT_GIF, images, duration=DURATION, loop=0)
        print("Done!")
    else:
        print("No frames generated.")

if __name__ == "__main__":
    create_gif()
