import os
import cv2
import torchaudio
import torch
from PIL import Image
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import load_and_transform_audio_data, load_and_transform_vision_data
from typing import List, Dict
import shutil

class CREMADProcessor:
    def __init__(self, video_dir, audio_dir, output_dir, frame_interval=30, device='cuda:6' if torch.cuda.is_available() else 'cpu'):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.device = device

        # os.makedirs(self.output_dir, exist_ok=True)

        # 加载 ImageBind 模型
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval().to(self.device)

    def extract_frames(self, video_path):
        """从视频中提取每隔 frame_interval 的一帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames

        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            index += 1
        cap.release()
        return frames

    def extract_frames_path(self, video_path, temp_dir="temp_frames"):
        os.makedirs(temp_dir, exist_ok=True)
        frames_paths = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames_paths

        index = 0
        frame_id = 0
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(temp_dir, f"{video_id}_frame{frame_id}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frames_paths.append(frame_path)
                frame_id += 1
            index += 1

        cap.release()
        return frames_paths

    def extract_label_from_filename(self, filename):
        """
        解析 CREMA-D 文件名格式：
        e.g., '1001_IEO_HAP_H_M.mp4'
        speaker_id_emotion_intensity_sentence_gender
        """
        stem = os.path.splitext(filename)[0] # 去掉文件扩展名
        parts = stem.split("_")
        return {
        "speaker_id": parts[0],
        "emotion": parts[1],
        "intensity": parts[2] if len(parts) > 2 else None,
        "sentence": parts[3] if len(parts) > 3 else None,
        "gender": parts[4] if len(parts) > 4 else None,
        }

    def process_all(self, save_file):
        """处理所有视频并将所有信息保存到一个 .pt 文件中"""
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".flv"))]
        print("Total video files found:", len(video_files))
        all_data = []

        for vid in tqdm(video_files, desc="Processing CREMA-D"):
            video_path = os.path.join(self.video_dir, vid)
            stem = os.path.splitext(vid)[0]
            audio_path = os.path.join(self.audio_dir, f"{stem}.wav")

            try:
                frames = self.extract_frames_path(video_path)
                if not frames:
                    print(f"skip：{vid}, cannot extract frames from video. Check if the video is valid.")
                    continue

                # 转换数据
                image_input = load_and_transform_vision_data(frames, self.device)
                audio_input = load_and_transform_audio_data([audio_path], self.device)

                # 提取嵌入
                with torch.no_grad():
                    image_embeddings = self.model({ModalityType.VISION: image_input})[ModalityType.VISION]
                    audio_embedding = self.model({ModalityType.AUDIO: audio_input})[ModalityType.AUDIO]

                    image_embedding = image_embeddings.mean(dim=0, keepdim=True)  # 平均池化图像帧

                # 标签信息
                label = self.extract_label_from_filename(vid)

                all_data.append({
                    "video": vid,
                    "image_embedding": image_embedding.cpu(),
                    "audio_embedding": audio_embedding.cpu(),
                    "label": label
                })

            except Exception as e:
                print(f"Failure {vid}：{e}")

        # 一次性保存所有样本
        torch.save(all_data, save_file)
        print(f" Save as {save_file}")
        shutil.rmtree("temp_frames")

if __name__ == "__main__":
    video_dir = "/data/zhouxiaokai/data/dataset/CREMA-D/VideoFlash/"
    audio_dir = "/data/zhouxiaokai/data/dataset/CREMA-D/AudioWAV/"
    output_dir = "/data/zhouxiaokai/codes/FedML/pretrained_embeddings/imagebind_crema_d_embeddings.pt"

    processor = CREMADProcessor(video_dir=video_dir, audio_dir=audio_dir, output_dir=output_dir, frame_interval=30)
    processor.process_all(output_dir)