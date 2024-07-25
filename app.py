import io
import numpy as np
import ffmpeg
import librosa
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from deepface import DeepFace
import streamlit as st
from collections import Counter
from PIL import Image

# Define emotion mapping
emotion_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5
}

# Define functions
def split_video_into_frames(video_bytes, frame_rate=1):
    video_stream = io.BytesIO(video_bytes)
    temp_video_path = '/tmp/temp_video.mp4'

    with open(temp_video_path, 'wb') as temp_file:
        temp_file.write(video_bytes)

    # Extract frames using ffmpeg
    frame_count = 0
    emotion_counter = Counter()

    try:
        probe = ffmpeg.probe(temp_video_path)
        duration = float(probe['format']['duration'])
        frame_interval = int(duration * 25)  # 25 frames per second
        for frame in ffmpeg.input(temp_video_path).output('pipe:', format='image2', vframes=1).run(capture_stdout=True, capture_stderr=True):
            img = Image.open(io.BytesIO(frame))
            img = np.array(img)

            if frame_count % frame_rate == 0:
                try:
                    analysis = DeepFace.analyze(img, actions=['emotion'])
                    if isinstance(analysis, list):
                        for result in analysis:
                            dominant_emotion = result['dominant_emotion']
                            emotion_counter[dominant_emotion] += 1
                    else:
                        dominant_emotion = analysis['dominant_emotion']
                        emotion_counter[dominant_emotion] += 1
                except Exception as e:
                    st.error(f"Error analyzing emotion: {str(e)}")

            frame_count += 1
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

    if emotion_counter:
        highest_occurring_emotion = emotion_counter.most_common(1)[0][0]
    else:
        highest_occurring_emotion = None

    return highest_occurring_emotion

def extract_audio_from_video(video_bytes):
    video_file = io.BytesIO(video_bytes)
    temp_video_path = '/tmp/temp_video.mp4'

    with open(temp_video_path, 'wb') as temp_file:
        temp_file.write(video_bytes)

    audio_file = io.BytesIO()
    ffmpeg.input(temp_video_path).output('pipe:', format='wav').run(capture_stdout=True, capture_stderr=True, stdout=audio_file)
    audio_file.seek(0)
    return audio_file.read()

def extract_features(audio_bytes, max_length=100):
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        features = np.vstack([mfccs, chroma, spectral_contrast])
        if features.shape[1] < max_length:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
        elif features.shape[1] > max_length:
            features = features[:, :max_length]
        return features.T
    except Exception as e:
        st.error(f"Error extracting features from audio: {str(e)}")
        return None

def main():
    st.title("Emotion Detection from Video")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        
        st.write("Processing video...please wait")
        highest_emotion = split_video_into_frames(video_bytes)
        audio_bytes = extract_audio_from_video(video_bytes)

        model_path = "SVMexec_modeltesting113.pkl"
        svm_model = joblib.load(model_path)
        scaler = joblib.load('scaler.pkl')

        features = extract_features(audio_bytes)
        if features is not None:
            features_2d = features.reshape(1, -1)
            features_normalized = scaler.transform(features_2d)

            predicted_class = svm_model.predict(features_normalized)[0]
            emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
            predicted_emotion = emotion_labels[predicted_class]

            if highest_emotion == predicted_emotion:
                st.write(f"The person in the video is {predicted_emotion}.")
            else:
                st.write(f"The emotions from the frames and audio do not match, but the facial expression seems to be {highest_emotion}, while the audio emotion seems to be {predicted_emotion}.")
        else:
            st.write("Failed to extract features from the audio file.")

if __name__ == "__main__":
    main()
