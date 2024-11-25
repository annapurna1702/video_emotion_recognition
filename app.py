import os
import numpy as np
import cv2
import librosa
import joblib
from deepface import DeepFace
import streamlit as st
from collections import Counter
from moviepy import VideoFileClip


emotion_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5
}


def split_video_into_frames_and_analyze_emotions(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    frame_count = 0
    success, frame = cap.read()

    emotion_counter = Counter()

    while success:
        if frame_count % frame_rate == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'])
                if isinstance(analysis, list):
                    for result in analysis:
                        dominant_emotion = result['dominant_emotion']
                        emotion_counter[dominant_emotion] += 1
                else:
                    dominant_emotion = analysis['dominant_emotion']
                    emotion_counter[dominant_emotion] += 1
            except Exception as e:
                pass

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    if emotion_counter:
        highest_occurring_emotion = emotion_counter.most_common(1)[0][0]
    else:
        highest_occurring_emotion = None

    return highest_occurring_emotion

def extract_audio_from_video(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    audio_array, sr = librosa.load(audio_path, sr=None)
    os.remove(audio_path)
    return audio_array, sr

def extract_features(audio_array, sr, max_length=100):
    try:
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sr)

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
    with open("style.css") as f:
      st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.title("Emotion Detection from Video")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.write("Processing video...please wait")
        highest_emotion = split_video_into_frames_and_analyze_emotions(video_path)
        audio_array, sr = extract_audio_from_video(video_path)

        model_path = "SVMexec_modeltesting113.pkl"
        svm_model = joblib.load(model_path)
        scaler = joblib.load('scaler.pkl')

        features = extract_features(audio_array, sr)
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
