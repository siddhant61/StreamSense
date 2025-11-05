# Dependency Management Guide

## Overview

StreamSense uses a two-tier dependency management approach:

1. **requirements.txt** - Development dependencies with version ranges
2. **requirements-lock.txt** - Production lock file with exact versions (generated)

## For Development

Install dependencies with flexible version ranges for development work:

```bash
pip install -r requirements.txt
```

This allows minor and patch updates within safe bounds while preventing breaking changes.

## For Production / Reproducible Environments

### Generating a Lock File

After installing dependencies successfully in your environment, generate an exact lock file:

```bash
# After pip install -r requirements.txt
pip freeze > requirements-lock.txt
```

### Using a Lock File

For reproducible production deployments, use the lock file:

```bash
pip install -r requirements-lock.txt
```

This ensures **exact** versions are installed, eliminating any variability.

## Dependency Categories

### Core Data & Scientific Computing
- **numpy**: Numerical computing foundation (pinned to 1.x)
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing algorithms
- **h5py**: HDF5 binary data format
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **bitstring**: Binary data parsing

### Neuroscience & Biometric Sensors
- **mne**: EEG data processing and analysis
- **muselsl**: Muse headband LSL streaming
- **pylsl**: Lab Streaming Layer Python bindings
- **pygatt**: Bluetooth LE GATT communication

### GUI & Visualization
- **PyQt5**: Qt5 GUI framework
- **vispy**: High-performance OpenGL visualizations
- **psychopy**: Psychophysics experiment framework

### Hardware Communication
- **pyserial**: Serial port communication
- **pybluez**: Bluetooth classic communication
- **pywifi**: WiFi device scanning

### Platform-Specific
- **wmi**: Windows Management Instrumentation (Windows only)

### Utilities
- **userpaths**: Cross-platform user directory paths

### Testing (Development Only)
- **pytest**: Testing framework
- **pytest-cov**: Code coverage plugin

## Version Pinning Strategy

We use **conservative version ranges**:

```python
package>=minimum_version,<next_major_version
```

**Example**: `numpy>=1.24.0,<2.0.0`

This means:
- ✓ Allows: numpy 1.24.0, 1.24.1, 1.25.0, 1.26.2
- ✗ Blocks: numpy 2.0.0, numpy 1.23.5

**Rationale**:
- Minimum version: Features we depend on
- Maximum version: Prevent breaking changes from major version bumps

## Updating Dependencies

### Updating a Single Package

```bash
pip install --upgrade "package>=new_version,<next_major"
pip freeze > requirements-lock.txt
```

### Updating All Packages

```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements-lock.txt
```

### Security Updates

Check for vulnerabilities:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Update vulnerable packages immediately:

```bash
pip install --upgrade vulnerable-package
pip freeze > requirements-lock.txt
```

## Platform-Specific Notes

### Windows
All dependencies should install without issues. The `wmi` package is Windows-only and will be automatically included.

### macOS / Linux
The `wmi` package will be skipped (platform condition in requirements.txt). Some features may be unavailable without platform-specific alternatives.

### Python Version Compatibility

**Supported**: Python 3.8, 3.9, 3.10, 3.11

**Testing Matrix**:
- Windows: Python 3.8, 3.10, 3.11
- macOS: Python 3.10, 3.11
- Linux: Python 3.9, 3.11

## Troubleshooting

### Dependency Conflicts

If you encounter version conflicts:

1. Check which package is causing the conflict:
   ```bash
   pip install -r requirements.txt --dry-run
   ```

2. Manually resolve by adjusting version ranges in requirements.txt

3. Consult package documentation for compatibility information

### Installation Failures

Common issues:

**PyQt5**: Requires system Qt libraries
- Windows: Installs automatically
- macOS: `brew install qt5`
- Linux: `apt-get install python3-pyqt5`

**pybluez**: Requires Bluetooth development libraries
- Windows: Included
- Linux: `apt-get install libbluetooth-dev`

**psychopy**: Large package with many dependencies
- May take 5-10 minutes to install
- Requires OpenGL support

## Security Best Practices

1. **Always use lock files in production**
   - Lock file = exact versions
   - Eliminates supply chain attack window

2. **Regularly scan for vulnerabilities**
   - Run `pip-audit` monthly
   - Update vulnerable packages immediately

3. **Never commit API keys or secrets**
   - Use environment variables
   - See: `archive/e4_basic_flow.py` (sanitized example)

4. **Verify package integrity**
   - pip automatically verifies checksums
   - Use `--require-hashes` for extra security

## Migration from Unpinned Dependencies

**Before** (October 2025 and earlier):
```txt
numpy
pandas
scipy
```
- 0% pinned
- Security risk: Any version could install
- Reproducibility risk: Different versions on different machines

**After** (November 2025):
```txt
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scipy>=1.10.0,<2.0.0
```
- 100% pinned with safe ranges
- Security: Known versions only
- Reproducibility: Controlled version ranges

## Further Reading

- [pip documentation](https://pip.pypa.io/en/stable/)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [pip-audit](https://pypi.org/project/pip-audit/)

---

**Last Updated**: November 2025
**Audit**: See `audit/nov_2025_comprehensive_audit.md`
