# Code Health Follow-up Tasks

## Typo Fix
- **Issue**: The Empatica streamer uses a misspelled method name `suscribe_to_data`, which propagates the typo to its callers and makes searchability/readability harder. 【F:streamer/stream_e4.py†L67-L93】【F:streamer/stream_e4.py†L226-L233】
- **Proposed Task**: Rename the method to `subscribe_to_data` and update every call site to use the corrected spelling.

## Bug Fix
- **Issue**: `StreamE4.start_streaming` joins the background `Process` immediately after receiving the "connected" signal, which blocks the caller forever while the child keeps streaming. 【F:streamer/stream_e4.py†L235-L241】
- **Proposed Task**: Keep a handle to the spawned process without joining it (or join only during shutdown) so that the API returns once the device is ready.

## Documentation Alignment
- **Issue**: The `Muse` constructor docstring documents callback parameters that are not part of the signature, creating confusion for integrators. 【F:helper/muse_helper.py†L26-L59】
- **Proposed Task**: Rewrite the docstring to describe the actual queue-based arguments or reintroduce the documented callbacks for consistency.

## Test Improvement
- **Issue**: `DataProcessor.detect_gaps` mutates its input DataFrame while performing complex gap detection, but there are no automated tests covering this behavior. 【F:data_processor.py†L178-L206】
- **Proposed Task**: Add unit tests that feed representative DataFrames (with time gaps, NaNs, and zero sequences) into `detect_gaps` to verify the returned mask and ensure unintended side effects are caught.
