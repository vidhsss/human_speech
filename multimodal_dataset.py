import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import cv2
from torchvision import transforms

class MultimodalSpeechDataset(Dataset):
    def __init__(self, csv_path, transform_audio=None, transform_video=None, max_frames=32):
        self.data = pd.read_csv(csv_path)
        self.transform_audio = transform_audio
        self.transform_video = transform_video
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, _ = torchaudio.load(row['audio_path'])  # [channels, time]
        video = self.load_video(row['video_path'])     # [frames, C, 224, 224]
        transcription = row['transcription']

        if self.transform_audio:
            audio = self.transform_audio(audio)

        return audio, video, transcription

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform_video:
                # print("Frame shape before transform:", frame.shape, type(frame))
                frame = self.transform_video(frame)  # [C, 224, 224]
            frames.append(frame)
            if len(frames) >= self.max_frames:
                break
        cap.release()
        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            pad = [frames[-1]] * (self.max_frames - len(frames))
            frames.extend(pad)
        else:
            frames = frames[:self.max_frames]
        video_tensor = torch.stack(frames)  # [frames, C, 224, 224]
        return video_tensor 