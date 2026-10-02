import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #e0e0f0;
        font-size: 1.05rem;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    .result-card-real {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .result-card-fake {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
    }

    .confidence-card {
        background-color: #1e2436;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .confidence-label {
        color: #ffffff;
        margin-bottom: 0.3rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .confidence-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('audio_spoof_detector_full.keras')

model = load_model()

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
MAX_LEN = SAMPLE_RATE * DURATION

def audio_to_melspectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    else:
        y = np.pad(y, (0, max(0, MAX_LEN - len(y))))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, y, sr

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Hero header ----------
st.markdown("""
<div class="hero">
    <h1>🎙️ Audio Deepfake Detector</h1>
    <p>CNN-based voice authenticity detection · Trained on ASVspoof 2019 LA · 99% Accuracy · 0.81% EER</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Analyze", "📊  Model Performance", "🕑  History"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Upload Audio")
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "flac", "mp3"], label_visibility="collapsed")

        if uploaded_file is not None:
            st.audio(uploaded_file)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            if st.button("🔎 Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio..."):
                    spec, y, sr = audio_to_melspectrogram(tmp_path)
                    spec_input = spec.reshape(1, spec.shape[0], spec.shape[1], 1)

                    prediction = model.predict(spec_input)[0][0]
                    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                    label = "Real Human Voice" if prediction > 0.5 else "AI-Generated / Spoofed Voice"

                    st.session_state.last_result = {
                        "label": label, "confidence": confidence,
                        "spec": spec, "y": y, "sr": sr, "filename": uploaded_file.name
                    }
                    st.session_state.history.append({
                        "File": uploaded_file.name, "Result": label, "Confidence": f"{confidence:.1f}%"
                    })

    with col2:
        st.markdown("#### Result")
        if "last_result" in st.session_state:
            r = st.session_state.last_result
            css_class = "result-card-real" if r["label"] == "Real Human Voice" else "result-card-fake"
            icon = "✅" if r["label"] == "Real Human Voice" else "❌"
            st.markdown(f'<div class="{css_class}">{icon} {r["label"]}</div>', unsafe_allow_html=True)
            st.write("")

            confidence_color = "#10b981" if r['confidence'] >= 80 else "#f59e0b" if r['confidence'] >= 60 else "#ef4444"
            st.markdown(f"""
            <div class="confidence-card" style="border: 2px solid {confidence_color};">
                <p class="confidence-label">CONFIDENCE SCORE</p>
                <p class="confidence-value" style="color: {confidence_color};">{r['confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.progress(int(r['confidence']))
        else:
            st.info("Upload and analyze an audio file to see results here.")

    if "last_result" in st.session_state:
        st.divider()
        st.markdown("#### Audio Analysis")
        r = st.session_state.last_result

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor('#0e1117')
        for ax in axs:
            ax.set_facecolor('#0e1117')

        librosa.display.waveshow(r["y"], sr=r["sr"], ax=axs[0], color='#7c3aed')
        axs[0].set_title("Waveform", color='white')
        axs[0].tick_params(colors='white')

        img = librosa.display.specshow(r["spec"], sr=r["sr"], x_axis='time', y_axis='mel', ax=axs[1], cmap='magma')
        axs[1].set_title("Mel Spectrogram", color='white')
        axs[1].tick_params(colors='white')
        fig.colorbar(img, ax=axs[1], format='%+2.0f dB')

        st.pyplot(fig)

with tab2:
    st.markdown("#### Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "99%")
    col2.metric("Equal Error Rate", "0.81%")
    col3.metric("Training Samples", "25,380")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset**: ASVspoof 2019 LA (Logical Access)")
        st.markdown("**Architecture**: CNN on Mel-spectrograms")
    with c2:
        st.markdown("**Class imbalance handling**: Class weighting (9:1 ratio)")
        st.markdown("**Framework**: TensorFlow / Keras")

    st.write("")
    st.markdown("#### Classification Report (Test Set)")
    report_data = {
        "Class": ["Spoof", "Bonafide"],
        "Precision": ["1.00", "0.97"],
        "Recall": ["1.00", "0.97"],
        "F1-Score": ["1.00", "0.97"]
    }
    st.table(pd.DataFrame(report_data))

    st.info("⚠️ Limitation: trained specifically on human speech. Non-speech audio inputs may produce unreliable results.")

with tab3:
    st.markdown("#### Analysis History (this session)")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.write("No files analyzed yet in this session.")