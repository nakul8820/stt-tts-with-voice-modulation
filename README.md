# STT-TTS with Voice Modulation 🎙️

A Python-based application designed to bridge the gap between Speech-to-Text (STT) and Text-to-Speech (TTS), featuring real-time voice modulation capabilities. This project aims to provide a modular framework for capturing audio, converting it to text, and regenerating it with specific vocal characteristics (pitch, speed, and tone).

> **⚠️ Work in Progress:** This project is currently under active development. Some features listed below may be partially implemented or are planned for future updates.

## 🚀 Overview

This repository provides a pipeline to:
1. **Speech-to-Text:** Transcribe live audio or pre-recorded files into text.
2. **Processing:** Allow for text manipulation or translation (optional).
3. **Voice Modulation:** Modify audio parameters to change the output voice profile.
4. **Text-to-Speech:** Convert the processed/modulated text back into high-quality audio.

## 🛠️ Tech Stack (Planned/Current)

*   **Language:** Python 3.x
*   **STT Engine:** (e.g., OpenAI Whisper, Google Speech Recognition)
*   **TTS Engine:** (e.g., pyttsx3, gTTS, or Coqui TTS)
*   **Audio Processing:** Librosa / Pydub (for modulation)
*   **Framework:** FastAPI (for API-based interaction)

## 📋 Features

*   [ ] Real-time microphone input processing.
*   [ ] Adjustable pitch and speed for voice modulation.
*   [ ] Support for multiple export formats (.wav, .mp3).
*   [ ] Clean CLI or Web Interface (Pending).

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nakul8820/stt-tts-with-voice-modulation.git](https://github.com/nakul8820/stt-tts-with-voice-modulation.git)
    cd stt-tts-with-voice-modulation
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate



## 🗺️ Roadmap

- [ ] Initial STT integration using Whisper.
- [ ] Basic Pitch shifting and Time-stretching logic.
- [ ] Integration of TTS for modulated output.
- [ ] Dockerization for easy deployment.

## 🤝 Contributing

Since this project is still in its early stages, contributions, suggestions, and bug reports are welcome! Feel free to open an issue or submit a pull request.
