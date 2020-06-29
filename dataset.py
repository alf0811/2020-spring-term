import random
import os
import glob
from PIL import Image
import cv2
import librosa
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AVCDataset(Dataset):

    def __init__(
            self,
            root,
            ):
        self.mp4_file_list = glob.glob(os.path.join(root, '*.mp4'))
        self.mp4_audio = [
                librosa.core.load(mp4_file, sr=44100) for mp4_file in self.mp4_file_list
                ]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __getitem__(self, _):
        idx = int(random.random() * len(self.mp4_file_list))
        sec = 0.5
        mp4_file = self.mp4_file_list[idx]
        video = cv2.VideoCapture(mp4_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        play_time = int(frame_num / fps) - sec - 1

        start_time = int(play_time * random.random())
        video.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)

        correct = random.random() > 0.5
        incorrect_start_time = 0
        master_audio, fs = self.mp4_audio[idx]

        if correct:
            start_frame = start_time*fs
            end_frame = int((start_time+sec)*fs)
            audio = master_audio[start_frame:end_frame]
        else:
            idx = int(random.random() * len(self.mp4_file_list))
            mp4_file = self.mp4_file_list[idx]
            video = cv2.VideoCapture(mp4_file)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
            play_time = int(frame_num / fps) - sec - 1

            start_time = int(play_time * random.random())

            master_audio, fs = self.mp4_audio[idx]
            incorrect_start_time = int(play_time * random.random())

            start_frame = incorrect_start_time*fs
            end_frame = int((incorrect_start_time+sec)*fs)
            audio = master_audio[start_frame:end_frame]

        audio = librosa.feature.melspectrogram(y=audio, sr=44100, n_mels=256)
        audio = audio[:, :256]
        audio = torch.Tensor(audio).unsqueeze(0)
        return image, audio, torch.LongTensor([correct]).squeeze()

    def __len__(self):
        return 10000
