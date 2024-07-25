import os
import numpy as np
import cv2
import librosa
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import streamlit as st
from collections import Counter

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
def split_video_into_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    frame_count = 0
    success, frame = cap.read()

    emotion_counter = Counter()

    while success:
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)

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
                st.error(f"")

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    if emotion_counter:
        highest_occurring_emotion = emotion_counter.most_common(1)[0][0]
        
    else:
        highest_occurring_emotion = None
        

    return highest_occurring_emotion

def extract_audio_from_video(video_path, audio_output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)
    video_clip.close()
    audio_clip.close()
    

def extract_features(file_path, max_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
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
        st.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

def main():
    st.title("Emotion Detection from Video")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        output_folder = 'output_frames'
        audio_output_path = 'output_audio.wav'

        st.write("Processing video...please wait")
        highest_emotion = split_video_into_frames(video_path, output_folder)
        extract_audio_from_video(video_path, audio_output_path)

        model_path = "SVMexec_modeltesting113.pkl"
        svm_model = joblib.load(model_path)
        scaler = joblib.load('scaler.pkl')

        features = extract_features(audio_output_path)
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
