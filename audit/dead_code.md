# Dead and Orphaned Code Analysis

This document outlines the findings of the dead and orphaned code analysis conducted as part of the project audit.

## Summary

The analysis revealed several Python scripts that are not integrated into the main `StreamSense` application flow initiated by `main.py`. While not "dead" in the sense that they are unreachable, they are "orphaned" and represent a disorganized collection of tools rather than a cohesive application.

## Orphaned Files

1.  **`helper/data_helper.py`**:
    -   **Description**: This file contains a number of utility functions for data manipulation, including `inspect_hdf5_file`, `clean_align_and_save_bad_data`, `convert_csv_to_hdf5`, and `convert_hdf5_to_csv`.
    -   **Status**: **Orphaned**. A `grep` search confirms that this file is never imported or used by any other script in the repository. Its functionality is entirely disconnected from the rest ofthe application.

2.  **`e4_basic_flow.py`**:
    -   **Description**: This script appears to be a self-contained example for connecting to the Empatica E4 streaming server and managing a device connection.
    -   **Status**: **Orphaned**. It is not imported or called by any other part of the application. It seems to be a developmental or testing script that was left in the repository.

3.  **`stream_info.py`**:
    -   **Description**: This script provides a useful utility for displaying a dashboard of all active LSL streams and their sampling rates. It uses `matplotlib` to generate the plot.
    -   **Status**: **Orphaned Utility**. While functional and related to the project's purpose, it is a standalone tool. It is not integrated into the main `main.py` CLI and must be run independently. The functionality it provides was presented as a key feature in the project's documentation (the `Logs/Slide24.JPG` image), but it was never integrated into the main user experience.
