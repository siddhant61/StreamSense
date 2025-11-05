# 📸 Screenshot Generation Guide

**Quick guide to generating professional UI screenshots for the README**

---

## 🎯 Purpose

The `scripts/capture_ui_screenshots.py` script automatically generates high-quality screenshots of the StreamSense UI for documentation purposes. These screenshots appear in:
- `README.md` (repository main page)
- `docs/UI_QUICK_START.md` (UI usage guide)
- Other documentation

---

## ⚡ Quick Start

### Prerequisites

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Specifically, you'll need:
- PyQt5
- pylsl
- All StreamSense dependencies

### Generate Screenshots

**Simply run:**

```bash
python scripts/capture_ui_screenshots.py
```

**What happens:**
1. UI window opens
2. Script automatically populates demo devices
3. Screenshots are captured (takes ~15 seconds)
4. Images saved to `docs/screenshots/`
5. Window closes automatically

**Output:**
```
docs/screenshots/
├── 01_initial_state.png
├── 02_devices_discovered.png
├── 03_device_connected.png
├── 04_multiple_devices.png
├── 05_lsl_streams_active.png
├── 06_recording_active.png
├── 07_recording_duration.png
├── 08_status_feedback.png
├── 09_full_window_overview.png
└── 10_device_cards_detail.png
```

---

## 📋 Screenshot Details

| # | Filename | Description | Shows |
|---|----------|-------------|-------|
| 1 | `01_initial_state.png` | Initial empty UI | Clean layout, discover button |
| 2 | `02_devices_discovered.png` | Devices found | Device cards, ready to connect |
| 3 | `03_device_connected.png` | Muse connected | Signal quality, connected status |
| 4 | `04_multiple_devices.png` | Multiple devices | Multi-device capability |
| 5 | `05_lsl_streams_active.png` | LSL streams | Live stream monitoring |
| 6 | `06_recording_active.png` | Recording started | Recording button, session ID |
| 7 | `07_recording_duration.png` | Recording with timer | Live duration counter |
| 8 | `08_status_feedback.png` | Status messages | Real-time feedback |
| 9 | `09_full_window_overview.png` | Full window | Complete UI overview |
| 10 | `10_device_cards_detail.png` | Device details | Device card close-up |

---

## 🔧 Customization

### Adding New Screenshots

Edit `scripts/capture_ui_screenshots.py`:

```python
def run_capture_sequence(self):
    # ... existing screenshots ...

    # Add your new screenshot here
    self.window.some_new_state()
    QApplication.processEvents()
    time.sleep(0.5)

    self.capture_screenshot(
        "11_my_new_state.png",
        "Description of the new state"
    )
```

### Changing Screenshot Quality

In `capture_screenshot()` method:

```python
pixmap.save(str(filepath), 'PNG', quality=100)  # 100 = maximum quality
```

### Adjusting Timing

If screenshots look incomplete, increase sleep time:

```python
time.sleep(0.5)  # Increase to 1.0 or 2.0 if needed
```

---

## 🎨 Screenshot Best Practices

### For Professional Results:

1. **Run on Windows** - UI looks best on Windows (native fonts, rendering)
2. **1920x1080+ display** - Ensures high-resolution screenshots
3. **Clean desktop** - Close other applications for clean taskbar
4. **Default theme** - Don't modify UI colors or fonts
5. **Good lighting** - N/A for screenshots, but matters if you photograph screen

### Image Specifications:

- **Format**: PNG (lossless)
- **Quality**: 100% (maximum)
- **Color**: 24-bit RGB
- **Compression**: PNG automatic compression
- **Size**: ~500KB - 2MB per image (depends on content)

---

## 🐛 Troubleshooting

### "No module named 'PyQt5'"

```bash
pip install PyQt5
```

### "Cannot connect to display"

You're on a headless server. Screenshots require a display server (X11, Wayland, etc.)

**Solutions:**
- Run on local machine with GUI
- Use Xvfb for headless screenshot capture:
  ```bash
  xvfb-run python scripts/capture_ui_screenshots.py
  ```

### Screenshots are blank or incomplete

- Increase sleep times in the script
- Check if PyQt5 is properly installed
- Try running UI manually first: `python ui/streamsense_ui.py`

### Script hangs or doesn't close

- Press Ctrl+C to force exit
- Check if QTimer.singleShot is working correctly
- Reduce wait times if testing

---

## 📤 After Generating Screenshots

### 1. Verify Screenshots

Check `docs/screenshots/` and open a few images to ensure they look good:

```bash
# On Windows
start docs\screenshots\09_full_window_overview.png

# On macOS
open docs/screenshots/09_full_window_overview.png

# On Linux
xdg-open docs/screenshots/09_full_window_overview.png
```

### 2. Update README

The README already references these screenshots. Once generated, they'll appear automatically:

```markdown
![Initial State](docs/screenshots/01_initial_state.png)
```

### 3. Commit Screenshots

```bash
git add docs/screenshots/
git commit -m "Add UI screenshots for documentation"
git push
```

### 4. View on GitHub

Go to your repository on GitHub and check the README to see the screenshots rendered.

---

## 🎯 When to Regenerate Screenshots

Regenerate screenshots when:
- ✅ UI design changes (colors, layout, fonts)
- ✅ New features are added to the UI
- ✅ Device cards or controls are modified
- ✅ Preparing for a release or presentation
- ✅ Screenshots look outdated

---

## 🚀 Quick Regeneration

**One-liner for updating everything:**

```bash
python scripts/capture_ui_screenshots.py && git add docs/screenshots/ && git commit -m "Update UI screenshots" && git push
```

---

## 💡 Tips for Job Interviews

1. **Show the script** - Demonstrates automation skills
2. **Explain the process** - Shows understanding of UI testing
3. **Mention reproducibility** - Screenshots can be regenerated anytime
4. **Highlight automation** - No manual screenshot capture needed

**Interview talking point:**
> "I automated the documentation screenshot process. The script programmatically simulates different UI states and captures high-quality images. This ensures screenshots are always up-to-date with the codebase and can be regenerated in seconds."

---

## 📚 Related Documentation

- [`scripts/README.md`](../scripts/README.md) - Scripts overview
- [`docs/UI_QUICK_START.md`](UI_QUICK_START.md) - UI usage guide
- [`README.md`](../README.md) - Main repository page

---

**Happy screenshot generating! 📸✨**

*Automated, professional, reproducible documentation.*
