# IML Project
This project uses machine learning to decide whether a person is allowed to enter based on their voice. It converts audio recordings into spectrograms and trains CNN models to classify speakers as authorized or unauthorized.

## Milestone 1
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



