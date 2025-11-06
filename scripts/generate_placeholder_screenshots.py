"""
Simple Screenshot Placeholder Generator

Creates placeholder screenshots for README when PyQt5 is not available.
This ensures the README doesn't have broken image links.

Usage:
    python scripts/generate_placeholder_screenshots.py

Requirements: None (uses only standard library)
"""

import os
from pathlib import Path


def create_svg_placeholder(filename, title, description, width=1400, height=900, color="#2d2d44"):
    """Create an SVG placeholder image."""

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{color}"/>

  <!-- Grid pattern -->
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#45475a" stroke-width="0.5" opacity="0.3"/>
    </pattern>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#grid)"/>

  <!-- Title -->
  <text x="{width/2}" y="{height/2 - 40}"
        font-family="Arial, sans-serif"
        font-size="48"
        font-weight="bold"
        fill="#cdd6f4"
        text-anchor="middle">
    {title}
  </text>

  <!-- Description -->
  <text x="{width/2}" y="{height/2 + 20}"
        font-family="Arial, sans-serif"
        font-size="24"
        fill="#a6adc8"
        text-anchor="middle">
    {description}
  </text>

  <!-- Filename -->
  <text x="{width/2}" y="{height/2 + 60}"
        font-family="monospace"
        font-size="18"
        fill="#7f849c"
        text-anchor="middle">
    {filename}
  </text>

  <!-- Watermark -->
  <text x="{width/2}" y="{height - 30}"
        font-family="Arial, sans-serif"
        font-size="16"
        fill="#585b70"
        text-anchor="middle"
        opacity="0.7">
    StreamSense Placeholder - Run 'python scripts/capture_ui_screenshots.py' for real screenshots
  </text>
</svg>
'''
    return svg_content


def main():
    """Generate placeholder screenshots."""

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  StreamSense Screenshot Placeholder Generator            ║")
    print("║  Creates SVG placeholders for README images              ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Create screenshots directory
    screenshots_dir = Path(__file__).parent.parent / "docs" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Define placeholder screenshots
    placeholders = [
        ("01_initial_state.png", "Initial State", "Clean UI ready for device discovery"),
        ("02_devices_discovered.png", "Devices Discovered", "Multiple devices found and listed"),
        ("03_device_connected.png", "Device Connected", "Muse headband connected with signal quality"),
        ("04_multiple_devices.png", "Multiple Devices", "Multiple devices streaming simultaneously"),
        ("05_lsl_streams_active.png", "LSL Streams Active", "Live LSL streams from all devices"),
        ("06_recording_active.png", "Recording Active", "Recording session in progress"),
        ("07_recording_duration.png", "Recording Duration", "Recording with live duration timer"),
        ("08_status_feedback.png", "Status Feedback", "Real-time status messages"),
        ("09_full_window_overview.png", "Full Window Overview", "Complete UI showing all features"),
        ("10_device_cards_detail.png", "Device Cards Detail", "Device cards with connection controls"),
    ]

    print(f"📁 Output directory: {screenshots_dir}\n")
    print("Generating placeholder screenshots...\n")

    for filename, title, description in placeholders:
        # Create SVG placeholder
        svg_content = create_svg_placeholder(filename, title, description)

        # Save as SVG (can be viewed in browser, converted to PNG later)
        svg_path = screenshots_dir / filename.replace('.png', '.svg')
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        print(f"✓ Created: {filename} → {svg_path.name}")

    print(f"\n✨ Created {len(placeholders)} placeholder screenshots!\n")

    print("📝 Next Steps:")
    print("   1. These are SVG placeholders - browsers and GitHub can display them")
    print("   2. To generate REAL screenshots:")
    print("      • Install dependencies: pip install PyQt5 pylsl")
    print("      • Run: python scripts/capture_ui_screenshots.py")
    print("   3. Update README.md image links to use .svg instead of .png (or convert to PNG)\n")

    print("💡 To convert SVG to PNG (optional):")
    print("   • Install: pip install cairosvg")
    print("   • Or use online converter: https://svgtopng.com/")
    print("   • Or use Inkscape: inkscape file.svg --export-png=file.png\n")


if __name__ == '__main__':
    main()
