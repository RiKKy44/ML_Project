# IML Project
This project uses machine learning to decide whether a person is allowed to enter based on their voice. It converts audio recordings into spectrograms and trains CNN models to classify speakers as authorized or unauthorized.

## Milestone 1

### Audio conversion
`wav_convereter.ps1` converts audio files to WAV, including files in all subfolders. Original files stay unchanged.

- **Installation:**

  Requires Windows 10/11. FFmpeg is downloaded automatically on first use.
- **Usage:**
  Eun directly in PowerShell:
  ```powershell
  .\wav_converter.ps1 -InputFolder "C:\recordings"
  ```
- **Output:**
  Converted files are saved in a `WAV` folder inside the selected folder, preserving the subfolder structure. Existing WAV files are skipped.

### Audio chunking
`smart_chunk_wav.py` splits WAV recordings into 1–3 second clips, using speech detection to skip long silences and prefer cuts at pauses.

- **Installation:** 

    Requires Python 3.10 or newer:
    ```
    python -m pip install numpy scipy soundfile webrtcvad-wheels
    ```
- **Usage:**
    ```
    python smart_chunk_wav.py <recordings> <chunks> --label 1
    ```
    - `recordings`: input WAV file or folder.
    - `chunks`: output folder, which must be new or empty.
    - `--label`: optional class assigned to every exported clip: `1` for allowed speakers, `0` for not allowed.
- **Output:**
    The output folder contains WAV clips, `manifest.csv` with labels and source timestamps, `skipped.csv` with reasons for skipped segments, and `run.json` with settings and statistics.

### Spectrogram creation
`create_spectrograms.py` converts the WAV chunks into log-Mel spectrograms for the CNN. Each chunk is loaded as 16 kHz mono, padded with silence to 3 seconds, and converted to a Mel spectrogram in dB (64 Mel bands, FFT size 1024, hop length 256). Every spectrogram has the shape `(64, 188)`.

- **Installation:**

    Requires Python 3.10 or newer:
    ```
    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install numpy soundfile librosa matplotlib
    ```
- **Usage:**
    ```
    python m1/create_spectrograms.py <spectrograms> --allowed <chunks_allowed> --not-allowed <chunks_not_allowed>
    ```
    - `spectrograms`: output folder, which must be new or empty.
    - `--allowed`: one or more chunk folders with class `1` clips.
    - `--not-allowed`: one or more chunk folders with class `0` clips.

    If a chunk folder has no `manifest.csv`, the recording name and `source_group` are read from the chunk folder names.
- **Output:**
    The output folder contains `npy/` with one `float32` array per chunk, `spectrograms.csv` with the label and all `manifest.csv` metadata for each array, `examples/` with a few spectrogram plots per class as PDF, and `run.json` with settings and statistics. `source_group` identifies the original recording, not the speaker. The `.npy` files and audio are ignored by Git. Example spectrograms of both classes are in `m1/figures/`.



