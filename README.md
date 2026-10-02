# 🎙️ Audio Deepfake Detection Dashboard

An interactive web app that detects whether a voice recording is real human speech or an AI-generated/spoofed voice, using a CNN trained on Mel-spectrograms.

🔗 **[Live Demo](https://audio-deepfake-detection-7n7r.onrender.com/)**

## Results
- Trained on the full ASVspoof 2019 LA dataset (25,380 samples)
- 99% accuracy, 0.81% Equal Error Rate (EER) on held-out test set
- Handled 9:1 class imbalance using class weighting

## Features
- Upload and analyze audio files (WAV, FLAC, MP3)
- Visual waveform and Mel-spectrogram display
- Confidence-scored predictions with color-coded results
- Session-based analysis history
- Model performance dashboard

## Tech Stack
Python, TensorFlow/Keras, librosa, Streamlit

## Limitations
- Trained specifically on human speech; non-speech audio inputs may produce unreliable results
- Generalization to AI voice generation methods outside the ASVspoof benchmark is not guaranteed
- Hosted on Render's free tier; cold starts and limited CPU may cause slower load/inference times compared to a production deployment

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```