import os
import re
import torch
import torchaudio
# import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from timm import create_model
from facenet_pytorch import MTCNN
import librosa
import tempfile
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, AutoModelForImageClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import timm
from django.views import View
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.utils import timezone
from django.db.models import F
from datetime import datetime
import uuid
import subprocess
import gc
from .models import VideoCount
import mediapipe as mp
import tensorflow as tf
import dlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import random
import sys
import parselmouth
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
import traceback
import whisper
from langdetect import detect
import pandas as pd
from faster_whisper import WhisperModel

# ReportLab for PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak 
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter

from collections import deque, Counter
from typing import Tuple, Deque, Dict, Any, Optional, List, Callable, Union

import logging
import time
import warnings

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Initialize Rich Console for progress bar and colored text
console = Console()

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils.text import slugify

from ultralytics import YOLO

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
from scipy.spatial import distance as dist
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def to_device(data, device):
    """Helper to move data to device with proper memory handling"""
    try:
        return data.to(device)
    except RuntimeError as e:
        #print(f"GPU memory error: {e}")
        clear_gpu_memory()
        return data.to('cpu') # Fallback to CPU

def free_gpu_memory(model):
    """Move model to CPU and clear CUDA cache."""
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    #print("Freed GPU memory.")

def move_model_to_device(model, device):
    """Move model to GPU with FP16 precision if available."""
    if torch.cuda.is_available():
        model = model.to(device).half()
    return model

def save_uploaded_video(request,video_file):
    """
    Saves the uploaded video in the media folder and returns its path.
    """
    try:
        # Ensure media/videos directory exists
        video_dir = os.path.join(settings.MEDIA_ROOT, "videos")
        os.makedirs(video_dir, exist_ok=True)

        video_uuid = request.session.get('video_uuid')

        # Generate a unique filename based on timestamp
        video_filename = f"video_{video_uuid}.mp4"
        video_path = os.path.join(video_dir, video_filename)

        # Save video file
        with open(video_path, "wb") as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        return os.path.join("videos", video_filename)  # Return relative path
    except Exception as e:
        #print(f"Error saving video: {e}")
        return None

def generate_filename(extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}.{extension}"

def extract_and_store_audio(video_path, request):
    """
    Extract audio from video and store in session and media storage
    
    Args:
        video_path (str): Path to video file
        request: Django request object
    
    Returns:
        str: Relative path to extracted audio file or None on failure
    """
    try:
        # Setup paths
        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        audio_dir = os.path.join(settings.MEDIA_ROOT, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"audio_{timestamp}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        # FFmpeg command to extract audio
        command = [
            'ffmpeg',
            '-i', video_absolute_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_path
        ]

        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            #print(f"FFmpeg error: {process.stderr.decode()}")
            return None

        # Store relative path for database/session storage
        audio_relative_path = os.path.join("audio", audio_filename)
        
        # Store in session
        request.session["extracted_audio_path"] = audio_relative_path
        request.session.modified = True

        return audio_relative_path

    except Exception as e:
        #print(f"Error extracting audio: {e}")
        return None

def cleanup_old_files(directory, max_age_hours=24):
    """Remove files older than max_age_hours from the specified directory"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (current_time - file_modified_time).total_seconds() > max_age_hours * 3600:
                try:
                    os.remove(filepath)
                    #print(f"Cleaned up old file: {filepath}")
                except OSError as e:
                    print(f"Error removing file {filepath}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

######################## deepfake ###################################


'''
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout_rate=0.3):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x).squeeze(-1)
        x = self.dropout2(x)
        return self.fc(x)

class LocalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure MTCNN is correctly initialized and available
        # It's better to pass the device from the main Deepfake class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.local_cnn = LocalCNN()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_lip_eye(self, face_img):
        h, w = face_img.shape[1], face_img.shape[2]
        lips = face_img[:, int(h*0.65):int(h*0.85), int(w*0.25):int(w*0.75)]
        eyes = face_img[:, int(h*0.2):int(h*0.45), int(w*0.2):int(w*0.8)]
        return lips.unsqueeze(0), eyes.unsqueeze(0)

    def forward(self, frame_np): # Expects a numpy array frame (BGR or RGB)
        # Convert numpy array to PIL Image for MTCNN
        frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)) # Assuming BGR input from OpenCV

        boxes, _ = self.mtcnn.detect(frame_pil)
        if boxes is None:
            return None # No face detected

        # Extract the largest face (MTCNN returns boxes in order of largest first if keep_all=False)
        x1, y1, x2, y2 = map(int, boxes[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_np.shape[1], x2), min(frame_np.shape[0], y2)

        face = frame_pil.crop((x1, y1, x2, y2))
        if face.size[0] == 0 or face.size[1] == 0:
            return None # Empty face crop

        face_tensor = self.transform(face).to(self.device) # Apply internal transform
        
        lips, eyes = self.extract_lip_eye(face_tensor) # lips/eyes are already tensors
        lips, eyes = lips.to(self.device), eyes.to(self.device)

        local_feat = torch.cat([
            self.local_cnn(lips),
            self.local_cnn(eyes)
        ], dim=1)
        
        global_feat = self.vit(face_tensor.unsqueeze(0)) # ViT expects batch dimension
        
        # Ensure global_feat is on the correct device. It should be from self.vit already
        return torch.cat([local_feat, global_feat], dim=1)

class DeepfakeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Assuming cnn_model is defined globally for SyncModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = timm.create_model('tf_efficientnet_b0', pretrained=True)
cnn_model.reset_classifier(0) # Resets classifier to 0 outputs, making it a feature extractor
cnn_model.eval().to(device)


class SyncModel(torch.nn.Module):
    def __init__(self, audio_feat_dim=768, visual_feat_dim=cnn_model.num_features):
        super().__init__()
        self.a_proj = torch.nn.Linear(audio_feat_dim, 128)
        self.v_proj = torch.nn.Linear(visual_feat_dim, 128)
        self.a_lstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.v_lstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.attn = torch.nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.classifier_bn_pool = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier_fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, audio, visual):
        a = self.a_proj(audio)
        v = self.v_proj(visual)
        a, _ = self.a_lstm(a)
        v, _ = self.v_lstm(v)
        fused, _ = self.attn(a, v, v)

        x = fused.transpose(1, 2)
        x = self.classifier_bn_pool(x).squeeze(-1)
        return torch.sigmoid(self.classifier_fc(x)).squeeze(-1)

# Ensure face_detector and facemark are loaded (from your original code)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
try:
    facemark_model_path = os.path.join(settings.BASE_DIR, 'models/lbfmodel.yaml')
    # Use resource_filename from pkg_resources if you need to load from an installed package
    # For local dev, direct path is fine.
    facemark.loadModel(facemark_model_path)
except Exception as e:
    print(f"Warning: Could not load facemark model at {facemark_model_path}. Lip sync check might be affected. Error: {e}")



# -------------------- deepfake detection class --------------------
class Deepfake(View):
    def __init__(self):
        super().__init__()

        self.MODEL_PATHS = {
            'audio': os.path.join(settings.BASE_DIR, 'models/new_deepfake/best_model_epoch_5.pth'),
            'visual_extractor': os.path.join(settings.BASE_DIR, 'models/new_deepfake/visual_feature_extractor.pth'),
            'classifier': os.path.join(settings.BASE_DIR, 'models/new_deepfake/deepfake_classifier.pth'),
            'lbf': os.path.join(settings.BASE_DIR, 'models/lbfmodel.yaml'),
            'sync_model': os.path.join(settings.BASE_DIR, 'models/new_deepfake/sync_model_opencv_hubert_pytorch.pth'),
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2.eval()
            self.wav2vec2.to(self.device)
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

            self.audio_model = AudioClassifier().to(self.device)
            state_dict_audio = torch.load(self.MODEL_PATHS['audio'], map_location=self.device)
            self.audio_model.load_state_dict(state_dict_audio, strict=False)
            self.audio_model.eval()

            self.extractor = VisualFeatureExtractor().to(self.device)
            state_dict_extractor = torch.load(self.MODEL_PATHS['visual_extractor'], map_location=self.device)
            self.extractor.load_state_dict(state_dict_extractor)
            self.extractor.eval()

            # Ensure the input_dim for DeepfakeClassifier is correct based on VisualFeatureExtractor's output
            # If VisualFeatureExtractor's output is 896, this is correct.
            self.model = DeepfakeClassifier(896).to(self.device)
            state_dict_classifier = torch.load(self.MODEL_PATHS['classifier'], map_location=self.device)
            self.model.load_state_dict(state_dict_classifier)
            self.model.eval()

            self.sync_model = SyncModel().to(self.device)
            state_dict_sync = torch.load(self.MODEL_PATHS['sync_model'], map_location=self.device)
            self.sync_model.load_state_dict(state_dict_sync)
            self.sync_model.eval()

            # NOTE: self.mtcnn here is used for initial face detection for display and cropping in predict_video
            # It's distinct from the MTCNN used inside VisualFeatureExtractor
            self.mtcnn = MTCNN(keep_all=False, device=self.device)

            # This transform was causing the original issue by being fed to DeepfakeClassifier directly
            # It's not directly used for the main DeepfakeClassifier now, as VisualFeatureExtractor handles its own transforms.
            # However, you might use it if you want to display the face crop at a specific size.
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)), # This size was for the classifier, now extractor takes care of it
                transforms.ToTensor(),
            ])


            self.LIP_LANDMARKS = list(range(48, 61))

            self.bundle = torchaudio.pipelines.HUBERT_BASE
            self.hubert_model = self.bundle.get_model().to(self.device).eval()
            self.hubert_sample_rate = self.bundle.sample_rate

            self.visual_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.segment_duration = 3

        except Exception as e:
            print(f"Error initializing Deepfake class: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Deepfake detector: {str(e)}")

    def calculate_scaled_dimensions(self, frame_width, frame_height, target_width, target_height):
        aspect_ratio = frame_width / frame_height
        target_aspect = target_width / target_height

        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            y_offset = (target_height - new_height) // 2
            return new_width, new_height, 0, y_offset
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            x_offset = (target_width - new_width) // 2
            return new_width, new_height, x_offset, 0

    def create_deepfake_bar_graph(self, real_count, fake_count, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)

        labels = ['Real', 'Fake']
        counts = [real_count, fake_count]
        colors = ['#39FF14', '#0000FF']

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        text_color = '#000000'
        grid_color = '#AAAAAA'

        plt.rcParams.update({
            'axes.facecolor': (0, 0, 0, 0),
            'axes.edgecolor': grid_color,
            'axes.labelcolor': text_color,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': grid_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'font.family': 'sans-serif',
            'font.weight': 'bold'
        })

        ax.bar(labels, counts, color=colors, alpha=0.8)
        ax.set_ylim(0, max(5, real_count + fake_count + 5))
        ax.set_title('Deepfake Detection', fontsize=6, fontweight='extra bold', color=text_color, pad=3)
        ax.set_ylabel('Frame Count', fontsize=6, color=text_color, fontweight='bold')
        ax.tick_params(axis='both', labelsize=5, colors=text_color)
        ax.tick_params(axis='x', labelrotation=0)
        ax.grid(True, alpha=0.15, color=grid_color)

        plt.tight_layout(rect=[0, 0, 1, 1])

        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        return graph_img

    def create_deepfake_pie_chart(self, real_count, fake_count, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)

        labels = ['Real', 'Fake']
        sizes = [real_count, fake_count] if (real_count + fake_count) > 0 else [1, 1]
        colors = ['#39FF14', '#0000FF']

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        text_color = '#000000'
        grid_color = '#AAAAAA'

        plt.rcParams.update({
            'axes.facecolor': (0, 0, 0, 0),
            'axes.edgecolor': grid_color,
            'axes.labelcolor': text_color,
            'axes.grid': False,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'font.family': 'sans-serif',
            'font.weight': 'bold'
        })

        if real_count == 0 and fake_count == 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct=lambda pct: 'No Data',
                            startangle=90, textprops={'fontsize': 5, 'color': text_color, 'fontweight': 'bold'},
                            radius=1.0)
        else:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                            startangle=90, textprops={'fontsize': 5, 'color': text_color, 'fontweight': 'bold'},
                            radius=1.0)

        ax.set_title('Real vs Fake Distribution', fontsize=6, fontweight='extra bold', color=text_color, pad=3)

        plt.tight_layout(rect=[0, 0, 1, 1])

        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        return graph_img

    def create_deepfake_line_graph(self, prediction_history, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)

        frames = list(range(1, len(prediction_history) + 1))
        real_preds = [1 if p == 0 else 0 for p in prediction_history]
        fake_preds = [1 if p == 1 else 0 for p in prediction_history]

        if any(real_preds) or any(fake_preds):
            ax.plot(frames, real_preds, label='Real', color='#39FF14', linewidth=1)
            ax.plot(frames, fake_preds, label='Fake', color='#FF0000', linewidth=1)
        else:
            ax.plot([], [], label='Real', color='#39FF14', linewidth=1)
            ax.plot([], [], label='Fake', color='#FF0000', linewidth=1)

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        text_color = '#000000'
        grid_color = '#AAAAAA'

        plt.rcParams.update({
            'axes.facecolor': (0, 0, 0, 0),
            'axes.edgecolor': grid_color,
            'axes.labelcolor': text_color,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': grid_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'font.family': 'sans-serif',
            'font.weight': 'bold'
        })

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Real', 'Fake'])
        ax.set_title('Prediction Trend Over Time', fontsize=6, fontweight='extra bold', color=text_color, pad=3)
        ax.set_xlabel('Frame Number', fontsize=6, color=text_color, fontweight='bold')
        ax.tick_params(axis='both', labelsize=5, colors=text_color)
        ax.grid(True, alpha=0.15, color=grid_color)
        ax.legend(fontsize=5, loc='upper right', facecolor='none', edgecolor='none')

        plt.tight_layout(rect=[0, 0, 1, 1])

        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        return graph_img

    def prepare_watermark(self, watermark_path, target_width, target_height):
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        if watermark is None:
            return None

        if watermark.shape[2] == 3:
            alpha = np.ones((watermark.shape[0], watermark.shape[1], 1), dtype=watermark.dtype) * 255
            watermark = cv2.merge([watermark, alpha])

        diagonal_length = int(np.sqrt(target_width**2 + target_height**2) * 0.4)
        aspect_ratio = watermark.shape[1] / watermark.shape[0]
        new_width = diagonal_length
        new_height = int(new_width / aspect_ratio)

        watermark_resized = cv2.resize(watermark, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        pad_size = max(new_width, new_height)
        padded = np.zeros((pad_size, pad_size, 4), dtype=np.uint8)
        x_offset = (pad_size - new_width) // 2
        y_offset = (pad_size - new_height) // 2

        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = watermark_resized

        angle = np.degrees(np.arctan2(target_height, target_width))
        matrix = cv2.getRotationMatrix2D((pad_size/2, pad_size/2), angle, 1.0)
        rotated = cv2.warpAffine(padded, matrix, (pad_size, pad_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        final_watermark = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        x_center_rotated = rotated.shape[1] // 2
        y_center_rotated = rotated.shape[0] // 2

        crop_x = x_center_rotated - target_width // 2
        crop_y = y_center_rotated - target_height // 2

        crop_x_start = max(0, crop_x)
        crop_y_start = max(0, crop_y)
        crop_x_end = min(rotated.shape[1], crop_x + target_width)
        crop_y_end = min(rotated.shape[0], crop_y + target_height)

        final_x_start = max(0, -crop_x)
        final_y_start = max(0, -crop_y)

        final_watermark[final_y_start : final_y_start + (crop_y_end - crop_y_start),
                                final_x_start : final_x_start + (crop_x_end - crop_x_start)] = rotated[
                                    crop_y_start:crop_y_end,
                                    crop_x_start:crop_x_end
                                ]
        return final_watermark

    def create_deepfake_graph_region(self, processed_frame_for_bg, graph_width, graph_height, bar_graph, pie_chart, line_graph, watermark_img=None):
        video_bg_source = cv2.resize(processed_frame_for_bg, (graph_width, graph_height))
        blurred_bg = cv2.GaussianBlur(video_bg_source, (15, 15), 10)

        gray_overlay = np.ones_like(blurred_bg) * 220
        blurred_overlay = cv2.GaussianBlur(gray_overlay, (49, 49), 15)

        blurred_bg = cv2.addWeighted(blurred_bg, 0.4, blurred_overlay, 0.6, 0)

        graph_region = blurred_bg.copy()

        padding = 15
        inner_graph_height = (graph_height - padding * 4) // 3
        inner_graph_width = graph_width - (padding * 2)

        inner_graph_height = max(1, inner_graph_height)
        inner_graph_width = max(1, inner_graph_width)

        y_start_bar = padding
        y_start_pie = padding + inner_graph_height + padding
        y_start_line = padding + inner_graph_height + padding + inner_graph_height + padding
        x_start = padding

        bar_graph_resized = cv2.resize(bar_graph, (inner_graph_width, inner_graph_height))
        if bar_graph_resized.shape[2] == 4:
            alpha_bar = bar_graph_resized[:, :, 3] / 255.0
            alpha_bar = np.dstack([alpha_bar] * 3)
            roi_bar = graph_region[y_start_bar:y_start_bar+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_bar.shape[:2] == bar_graph_resized.shape[:2]:
                graph_region[y_start_bar:y_start_bar+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_bar * (1 - alpha_bar) +
                    bar_graph_resized[:, :, :3] * alpha_bar
                ).astype(np.uint8)

        pie_chart_resized = cv2.resize(pie_chart, (inner_graph_width, inner_graph_height))
        if pie_chart_resized.shape[2] == 4:
            alpha_pie = pie_chart_resized[:, :, 3] / 255.0
            alpha_pie = np.dstack([alpha_pie] * 3)
            roi_pie = graph_region[y_start_pie:y_start_pie+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_pie.shape[:2] == pie_chart_resized.shape[:2]:
                graph_region[y_start_pie:y_start_pie+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_pie * (1 - alpha_pie) +
                    pie_chart_resized[:, :, :3] * alpha_pie
                ).astype(np.uint8)

        line_graph_resized = cv2.resize(line_graph, (inner_graph_width, inner_graph_height))
        if line_graph_resized.shape[2] == 4:
            alpha_line = line_graph_resized[:, :, 3] / 255.0
            alpha_line = np.dstack([alpha_line] * 3)
            roi_line = graph_region[y_start_line:y_start_line+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_line.shape[:2] == line_graph_resized.shape[:2]:
                graph_region[y_start_line:y_start_line+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_line * (1 - alpha_line) +
                    line_graph_resized[:, :, :3] * alpha_line
                ).astype(np.uint8)

        if watermark_img is not None:
            watermark_sized = cv2.resize(watermark_img, (graph_width, graph_height))
            if watermark_sized.shape[2] == 4:
                alpha_w = watermark_sized[:, :, 3] / 255.0
                alpha_w = np.dstack([alpha_w] * 3)
                graph_region = (graph_region * (1 - alpha_w * 0.4) +
                                watermark_sized[:, :, :3] * (alpha_w * 0.4)).astype(np.uint8)

        return graph_region

    def create_bordered_frame(self, main_content, heading_texts, output_width, output_height, heading_height, frame_width, real_count=0, fake_count=0):
        final_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        actual_titlebar_height = 40
        titlebar = np.ones((actual_titlebar_height, output_width, 3), dtype=np.uint8) * 255

        gradient = np.linspace(0.97, 1.0, actual_titlebar_height)[:, np.newaxis, np.newaxis]
        titlebar = (titlebar * gradient).astype(np.uint8)

        cv2.line(titlebar, (0, actual_titlebar_height-1), (output_width, actual_titlebar_height-1),
                 (240, 240, 240), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        text_color = (70, 70, 70)
        shadow_color = (200, 200, 200)
        text_thickness = 2

        temp_text_size = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_y = (actual_titlebar_height + temp_text_size[1]) // 2

        text_size_orig = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_x_orig = (frame_width - text_size_orig[0]) // 2

        cv2.putText(titlebar, heading_texts[0],
                            (text_x_orig+1, text_y+1),
                            font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[0],
                            (text_x_orig, text_y),
                            font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        text_size_foai = cv2.getTextSize(heading_texts[1], font, font_scale, text_thickness)[0]
        processed_panel_width = int(frame_width * 0.7)
        # right_panels_start_x = frame_width + processed_panel_width # This variable is not used
        # right_panels_width = output_width - right_panels_start_x # This variable is not used
        text_x_foai = frame_width + (processed_panel_width - text_size_foai[0]) // 2

        cv2.putText(titlebar, heading_texts[1],
                            (text_x_foai+1, text_y+1),
                            font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[1],
                            (text_x_foai, text_y),
                            font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        graph_panel_width = output_width - frame_width - processed_panel_width
        analysis_panel_start_x = frame_width + processed_panel_width

        font_size_metrics = 0.35
        text_thickness_metrics = 1

        title_padding = 5
        line_spacing = actual_titlebar_height / 3

        text_y1 = int(title_padding + line_spacing)
        text_y2 = int(title_padding + (line_spacing * 2))

        total_frames = max(1, real_count + fake_count)
        real_percent = (real_count / total_frames) * 100
        fake_percent = (fake_count / total_frames) * 100

        metric_area_width = graph_panel_width
        metric_padding = int(metric_area_width * 0.05)
        metric_x = analysis_panel_start_x + metric_padding

        metric_color = (255, 0, 0)

        cv2.putText(titlebar, f"Real: {real_percent:.1f}%",
                            (metric_x+1, text_y1+1),
                            font, font_size_metrics, shadow_color,
                            text_thickness_metrics, cv2.LINE_AA)
        cv2.putText(titlebar, f"Real: {real_percent:.1f}%",
                            (metric_x, text_y1),
                            font, font_size_metrics, metric_color,
                            text_thickness_metrics, cv2.LINE_AA)

        cv2.putText(titlebar, f"Fake: {fake_percent:.1f}%",
                            (metric_x+1, text_y2+1),
                            font, font_size_metrics, shadow_color,
                            text_thickness_metrics, cv2.LINE_AA)
        cv2.putText(titlebar, f"Fake: {fake_percent:.1f}%",
                            (metric_x, text_y2),
                            font, font_size_metrics, metric_color,
                            text_thickness_metrics, cv2.LINE_AA)

        final_frame[0:actual_titlebar_height, :] = titlebar
        content_y = actual_titlebar_height + 5
        content_height, content_width = main_content.shape[:2]
        final_frame[content_y:content_y+content_height, :content_width] = main_content

        return final_frame

    def extract_audio_ffmpeg(self, video_path, temp_audio_path):
        try:
            command = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-y', temp_audio_path
            ]
            process = subprocess.run(command, capture_output=True)
            if process.returncode != 0:
                print(f"FFmpeg audio extraction failed: {process.stderr.decode()}")
                return None
            return temp_audio_path
        except Exception as e:
            print(f"Error during FFmpeg audio extraction: {e}")
            return None

    def merge_video_audio_ffmpeg(self, video_path, audio_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            process = subprocess.run(command, capture_output=True)
            if process.returncode != 0:
                print(f"FFmpeg video-audio merge failed: {process.stderr.decode()}")
                # If merging fails, keep the video-only output and report
                if os.path.exists(video_path):
                    os.rename(video_path, output_path) # Rename original video to output_path
                return False
            return True
        except Exception as e:
            print(f"Error during FFmpeg video-audio merge: {e}")
            if os.path.exists(video_path):
                os.rename(video_path, output_path) # Rename original video to output_path
            return False

    def extract_audio_from_video_v1(self, video_path, audio_out_path="extracted_audio.wav"):
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_out_path, codec='pcm_s16le')
            video.audio.close()
            video.close()
            return audio_out_path
        except Exception as e:
            print(f"Error in extract_audio_from_video_v1: {e}")
            return None

    def split_audio(self, audio_path, chunk_duration=3, target_sr=16000):
        if not audio_path or not os.path.exists(audio_path):
            return [], target_sr
        try:
            wav, sr = librosa.load(audio_path, sr=target_sr)
            chunk_len = int(chunk_duration * target_sr)
            chunks = [wav[i:i+chunk_len] for i in range(0, len(wav), chunk_len) if len(wav[i:i+chunk_len]) == chunk_len]
            return chunks, target_sr
        except Exception as e:
            print(f"Error in split_audio: {e}")
            return [], target_sr

    def predict_chunks(self, chunks, processor, wav2vec2, model, device):
        model.eval()
        preds = []
        for chunk in chunks:
            try:
                inputs = processor(chunk, return_tensors="pt", sampling_rate=16000, padding=True).to(device)
                with torch.no_grad():
                    features = wav2vec2(inputs.input_values).last_hidden_state
                    outputs = model(features)
                    pred = outputs.argmax(dim=1).item()
                    preds.append(pred)
            except Exception as e:
                print(f"Error predicting audio chunk: {e}")
                preds.append(-1)
        return preds

    def classify_video(self, video_path, processor, wav2vec2, model, device):
        audio_path = self.extract_audio_from_video_v1(video_path)
        if not audio_path:
            return "Undetermined"
        chunks, sr = self.split_audio(audio_path)
        if not chunks:
            os.remove(audio_path)
            return "Undetermined"

        preds = self.predict_chunks(chunks, processor, wav2vec2, self.audio_model, device)
        os.remove(audio_path)

        valid_preds = [p for p in preds if p != -1]
        if not valid_preds:
            return "Undetermined"

        fake_count = valid_preds.count(1)
        fake_ratio = fake_count / len(valid_preds)

        label = "Fake" if fake_ratio >= 0.2 else "Real"
        return label

    def extract_audio_from_video_v2(self, video_path):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                audio_path = tmpfile.name
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=self.hubert_sample_rate, verbose=False, logger=None)
            clip.audio.close()
            clip.close()
            return audio_path
        except Exception as e:
            print(f"Error in extract_audio_from_video_v2: {e}")
            return None

    def split_segments(self, array, segment_len):
        if len(array) == 0 or segment_len <= 0:
            return []
        return [array[i:i+segment_len] for i in range(0, len(array), segment_len) if len(array[i:i+segment_len]) == segment_len]
    
    def extract_audio_features_video(self, video_path):
        try:
            audio_path = self.extract_audio_from_video_v2(video_path)
            if not audio_path:
                return []
            
            waveform, sr = torchaudio.load(audio_path)
            os.remove(audio_path)  # Clean up temp audio file
            
            if sr != self.hubert_sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.hubert_sample_rate)(waveform)

            with torch.no_grad():
                # Ensure waveform is 1-channel for Hubert
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                feats = self.hubert_model(waveform.to(self.device))[0][0].cpu().numpy()

            seg_len = int(self.hubert_sample_rate / 320) * self.segment_duration
            return self.split_segments(feats, seg_len)
            
        except Exception as e:
            print(f"Error extracting audio features for sync model: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_visual_features_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file for visual feature extraction (sync model).")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        lip_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Using face_detector from init for lip sync detection
                faces = face_detector.detectMultiScale(gray, 1.1, 5) 
                if not len(faces):
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if no face
                    continue
                face = faces[0] # Take the first detected face

                # Using facemark from init for lip sync detection
                ok, landmarks = facemark.fit(gray, np.array([face]))
                if not ok:
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if no landmarks
                    continue

                shape = landmarks[0][0]
                lip_pts = shape[self.LIP_LANDMARKS]
                x1, y1 = np.min(lip_pts, axis=0).astype(int)
                x2, y2 = np.max(lip_pts, axis=0).astype(int)
                pad = 5
                crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                                 max(0,x1-pad):min(frame.shape[1],x2+pad)]
                if crop.size == 0:
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if empty crop
                    continue

                tensor = self.visual_transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = cnn_model.forward_features(tensor)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1))
                    lip_features.append(pooled.view(-1).cpu().numpy())
            except Exception as e:
                print(f"Error processing visual frame for lip sync: {e}")
                import traceback
                traceback.print_exc()
                lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros on error

        cap.release()
        seg_len = int(fps * self.segment_duration)
        return self.split_segments(np.array(lip_features), seg_len)


    def predict_sync(self, video_path, threshold=0.5):
        audio_segs = self.extract_audio_features_video(video_path)
        visual_segs = self.extract_visual_features_from_video(video_path)
        
        if not audio_segs or not visual_segs:
            print("Warning: Missing audio or visual segments for lip-sync prediction.")
            return "Undetermined", 0.0

        scores = []
        with torch.no_grad():
            min_len = min(len(audio_segs), len(visual_segs))
            for i in range(min_len):
                a_seg = audio_segs[i]
                v_seg = visual_segs[i]

                if len(a_seg) == 0 or len(v_seg) == 0:
                    print(f"Skipping sync segment {i} due to empty features.")
                    continue

                a_t = torch.tensor(a_seg, dtype=torch.float32).unsqueeze(0).to(self.device)
                v_t = torch.tensor(v_seg, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Check shapes before passing to sync_model.
                # a_t should be [1, sequence_length, audio_feature_dim (768)]
                # v_t should be [1, sequence_length, visual_feature_dim (cnn_model.num_features)]
                # self.sync_model expects [batch_size, seq_len, feature_dim]
                # If a_seg is [seq_len, 768] and v_seg is [seq_len, cnn_model.num_features]
                # Then unsqueeze(0) will make them [1, seq_len, feature_dim]
                
                if a_t.shape[1] > 0 and v_t.shape[1] > 0:
                    try:
                        scores.append(self.sync_model(a_t, v_t).item())
                    except Exception as e:
                        print(f"Error during sync model inference for segment {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Decide how to handle this error: append a default score, or skip?
                        # For now, it will just not append to scores, making it robust.
                else:
                    print(f"Skipping sync segment {i} due to zero sequence length after tensor conversion.")


        if not scores:
            print("No valid scores obtained for lip-sync prediction.")
            return "Undetermined", 0.0

        avg_score = float(np.mean(scores))
        label = "Fake" if avg_score > threshold else "Real"
        return label, avg_score

    def label_to_prob(self, label):
        return 1.0 if label == "Fake" else 0.0

    def predict_video(self, video_path, output_path):
        if self.model is None:
            return {"error": "Deepfake detection model not loaded."}

        report_data = [['Frame No.', 'Timestamp', 'Visual Verdict', 'Visual Confidence', 'Audio Verdict', 'Lip-Sync Verdict', 'Overall Weighted Verdict']]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open video file at {video_path}."}

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate panel widths based on original frame width
        processed_panel_width = int(fw * 0.7)
        graph_panel_width = int(fw * 0.3)
        # Ensure minimum sizes to avoid zero-dimension issues
        if fw < 100: # Arbitrary small width check
            processed_panel_width = max(100, processed_panel_width) # Ensure at least 100px
            graph_panel_width = max(50, graph_panel_width) # Ensure at least 50px

        output_width = fw + processed_panel_width + graph_panel_width
        output_width += output_width % 2 # Ensure even width for video writers

        heading_height = 45 if 45 % 2 == 0 else 46 # Ensure even height
        output_height = fh + heading_height
        output_height += output_height % 2 # Ensure even height

        graph_width = graph_panel_width # Corrected graph width
        graph_height = fh # Graphs will take up the full height of the video frame area

        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_audio_file_for_merge = os.path.join(temp_dir, f"temp_audio_merge_{timestamp_str}.wav")

        audio_path_for_merge = self.extract_audio_ffmpeg(video_path, temp_audio_file_for_merge)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard H.264 compatible codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        visual_real_count, visual_fake_count = 0, 0
        frame_count = 0
        visual_prediction_history = []
        last_box = None
        last_visual_prediction = -1 # Initialize to -1 (undetermined)

        watermark_path = os.path.join(settings.STATIC_ROOT, "testapp", "images","watermark.png")
        watermark = self.prepare_watermark(watermark_path, graph_width, graph_height)

        print("\n--------------------Audio Check---------------------\n")
        pred_audio = self.classify_video(video_path, self.processor, self.wav2vec2, self.audio_model, self.device)
        print(f"Audio-based prediction: {pred_audio}")

        print("\n--------------------Lip Sync Check---------------------\n")
        pred_sync, sync_score = self.predict_sync(video_path)
        print(f"Lip-sync based prediction: {pred_sync} (Score: {sync_score:.4f})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_frame = time.strftime('%H:%M:%S', time.gmtime(frame_count/fps))
            original_frame = frame.copy()
            annotated_frame = frame.copy() # Frame to draw detections on

            current_visual_prediction = -1 # Reset for each frame
            current_box = None
            visual_confidence_value = 0.0

            try:
                # Use VisualFeatureExtractor directly for classification
                # It handles its own MTCNN, cropping, resizing, and feature extraction
                visual_features = self.extractor.forward(original_frame) # extractor takes numpy frame (BGR assumed)

                if visual_features is not None:
                    # visual_features is already a tensor [1, 896]
                    with torch.no_grad():
                        output = self.model(visual_features) # DeepfakeClassifier expects features (896-dim)
                        prediction = torch.argmax(output, dim=1).item()
                        visual_confidence_value = torch.softmax(output, dim=1)[0][prediction].item() * 100
                        current_visual_prediction = prediction

                    # Get the box from the extractor's last detection for drawing
                    # This assumes extractor has a way to store/retrieve the last detected box
                    # For simplicity, let's re-run MTCNN on the original frame for bounding box display
                    # This avoids modifying VisualFeatureExtractor internal state unnecessarily for just display.
                    temp_img_for_box = Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                    box_for_display, conf_for_display = self.mtcnn.detect(temp_img_for_box)
                    if box_for_display is not None and conf_for_display is not None and conf_for_display[0] > 0.9:
                        x1, y1, x2, y2 = map(int, box_for_display[0])
                        current_box = (max(0, x1), max(0, y1), min(original_frame.shape[1], x2), min(original_frame.shape[0], y2))
                    else:
                        current_box = None # If MTCNN for display fails, no box for this frame


            except Exception as e:
                # IMPORTANT: Print the error to debug!
                print(f"Error during visual deepfake classification at frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                # If an error occurs, current_visual_prediction remains -1, which is handled
                # This leads to "Detecting..." or "No Face" on the display
                current_visual_prediction = -1 # Ensure it's explicitly -1 on error
                current_box = None # Clear box on error

            # Update last_box and last_visual_prediction for continuity
            if current_box is not None and current_visual_prediction != -1: # Only update if a valid prediction was made for this frame
                last_box = current_box
                last_visual_prediction = current_visual_prediction
            elif last_box is not None: # If no face or error this frame, but had a previous one, use it for display continuity
                current_box = last_box
                current_visual_prediction = last_visual_prediction # Use last valid prediction for display

            visual_prediction_history.append(current_visual_prediction)

            if current_visual_prediction == 0:
                visual_real_count += 1
            elif current_visual_prediction == 1:
                visual_fake_count += 1
            
            # Determine visual verdict for report based on the current frame's prediction
            visual_verdict_for_report = "Real" if current_visual_prediction == 0 else ("Fake" if current_visual_prediction == 1 else "No Face")
            
            visual_p_current = self.label_to_prob(visual_verdict_for_report)
            audio_p_overall = self.label_to_prob(pred_audio) # Overall audio verdict
            sync_p_overall = self.label_to_prob(pred_sync)   # Overall lip-sync verdict
            
            # Calculate overall weighted verdict for THIS FRAME (using overall audio/sync)
            combined_score_for_report_frame = visual_p_current * 0.3 + audio_p_overall * 0.2 + sync_p_overall * 0.5
            overall_weighted_verdict_for_report_frame = "Fake" if combined_score_for_report_frame >= 0.5 else "Real"

            report_data.append([
                str(frame_count),
                timestamp_frame,
                visual_verdict_for_report,
                f"{visual_confidence_value:.2f}%" if current_visual_prediction != -1 else "N/A",
                pred_audio, # Audio verdict is overall for the video
                pred_sync,  # Lip-sync verdict is overall for the video
                overall_weighted_verdict_for_report_frame
            ])

            # Drawing on the annotated_frame
            if current_box is not None:
                x1, y1, x2, y2 = current_box
                verdict_display_text = "Real" if current_visual_prediction == 0 else ("Fake" if current_visual_prediction == 1 else "No Face (Internal Error)") # Changed from "Detecting..."
                color = (0, 255, 0) if current_visual_prediction == 0 else ((0, 0, 255) if current_visual_prediction == 1 else (255, 255, 0)) # Yellow for "No Face"
                
                # Draw confidence if available
                confidence_text = f" Conf: {visual_confidence_value:.1f}%" if current_visual_prediction != -1 else ""

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f'Verdict: {verdict_display_text}{confidence_text}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                # This condition now correctly means no face was found or processed by VisualFeatureExtractor
                cv2.putText(annotated_frame, 'No Face Detected For Analysis', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA) # Orange for "No Face"

            # Resize annotated frame for processed panel display
            new_w_proc, new_h_proc, x_offset_proc, y_offset_proc = self.calculate_scaled_dimensions(
                fw, fh, processed_panel_width, fh
            )
            annotated_frame_resized = cv2.resize(annotated_frame, (new_w_proc, new_h_proc))
            processed_panel_frame = np.zeros((fh, processed_panel_width, 3), dtype=np.uint8)
            processed_panel_frame[y_offset_proc : y_offset_proc + new_h_proc,
                                        x_offset_proc : x_offset_proc + new_w_proc] = annotated_frame_resized

            # Graph generation
            graph_height_per_chart = graph_height // 3
            bar_graph_img = self.create_deepfake_bar_graph(visual_real_count, visual_fake_count, graph_width, graph_height_per_chart)
            pie_chart_img = self.create_deepfake_pie_chart(visual_real_count, visual_fake_count, graph_width, graph_height_per_chart)
            line_graph_img = self.create_deepfake_line_graph(visual_prediction_history, graph_width, graph_height_per_chart)

            graph_region = self.create_deepfake_graph_region(
                processed_panel_frame, # Use processed frame for background
                graph_width,
                graph_height,
                bar_graph_img,
                pie_chart_img,
                line_graph_img,
                watermark
            )

            # Concatenate panels horizontally
            main_content = cv2.hconcat([original_frame, processed_panel_frame, graph_region])

            # Create final frame with title bar
            final_frame = self.create_bordered_frame(
                main_content,
                ["Original Video", "Processed Output (FOAI)", "Analysis"],
                output_width,
                output_height,
                heading_height,
                fw,
                real_count=visual_real_count,
                fake_count=visual_fake_count
            )

            out.write(final_frame)

        cap.release()
        out.release()

        # Final verdicts for the summary report
        visual_p_final = self.label_to_prob("Fake" if visual_fake_count > visual_real_count else "Real")
        audio_p_final = self.label_to_prob(pred_audio)
        sync_p_final = self.label_to_prob(pred_sync)
        
        combined_score_final = visual_p_final * 0.3 + audio_p_final * 0.2 + sync_p_final * 0.5
        final_verdict_summary = "Fake" if combined_score_final >= 0.5 else "Real"

        if visual_real_count == 0 and visual_fake_count == 0:
            final_verdict_summary = "Undetermined (No faces detected for visual analysis)"

        report_path = os.path.join(settings.MEDIA_ROOT, f"reports/deepfake_report_{timestamp_str}.pdf")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Deepfake Detection Report", styles['Heading1']))
        elements.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Overall Weighted Verdict: <b>{final_verdict_summary}</b>", styles['Normal']))
        elements.append(Paragraph(f"Visual (Frame-based) Analysis: Real Frames: {visual_real_count}, Fake Frames: {visual_fake_count}", styles['Normal']))
        elements.append(Paragraph(f"Audio-based Prediction: {pred_audio}", styles['Normal']))
        elements.append(Paragraph(f"Lip-Sync Prediction: {pred_sync} (Average Score: {sync_score:.4f})", styles['Normal']))
        elements.append(Paragraph("<br/><br/>", styles['Normal']))
        
        table = Table(report_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        
        doc.build(elements)

        if audio_path_for_merge and os.path.exists(audio_path_for_merge):
            temp_video_with_audio_merged = os.path.join(temp_dir, f"temp_merged_video_{timestamp_str}.mp4")
            merge_success = self.merge_video_audio_ffmpeg(output_path, audio_path_for_merge, temp_video_with_audio_merged)
            if merge_success:
                os.remove(output_path)
                os.rename(temp_video_with_audio_merged, output_path)
            else:
                print(f"Failed to merge audio. Video saved without audio at {output_path}")
        else:
            print(f"No audio extracted or audio file not found. Video saved without audio at {output_path}")

        try:
            if audio_path_for_merge and os.path.exists(audio_path_for_merge):
                os.remove(audio_path_for_merge)
        except Exception as e:
            print(f"Error cleaning up temporary audio file: {e}")
            pass

        return {
            'verdict': final_verdict_summary,
            'report_path': f"{settings.MEDIA_URL}reports/deepfake_report_{timestamp_str}.pdf",
            'audio_verdict': pred_audio,
            'lip_sync_verdict': pred_sync,
            'visual_verdict': "Fake" if visual_fake_count > visual_real_count else "Real" if (visual_real_count + visual_fake_count) > 0 else "Undetermined"
        }

    def encode_video(self, input_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding failed: {process.stderr.decode()}")
                return None

            return output_path

        except Exception as e:
            print(f"Error during video encoding: {e}")
            return None

    def post(self, request):
        if self.model is None:
            return JsonResponse({"error": "Deepfake detection model failed to load."}, status=500)

        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            video_uuid = request.session.get("video_uuid")
            if not video_uuid:
                return JsonResponse({"error": "No video UUID found in session"}, status=400)

            output_dir = os.path.join(settings.MEDIA_ROOT, "output_videos")
            os.makedirs(output_dir, exist_ok=True)

            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # final_output_name_encoded = f"deepfake_analysis_{timestamp}.mp4"
            # temp_output_name_encoded = f"temp_deepfake_analysis_{timestamp}.mp4"
           
            final_output_name_encoded = f"deepfake_analysis_{video_uuid}.mp4"
            temp_output_name_encoded = f"temp_deepfake_analysis_{video_uuid}.mp4"

            temp_output_path_encoded = os.path.join(output_dir, temp_output_name_encoded)
            final_output_path_encoded = os.path.join(output_dir, final_output_name_encoded)


            verdict_data = self.predict_video(video_absolute_path, temp_output_path_encoded)
            encoded_path = self.encode_video(temp_output_path_encoded, final_output_path_encoded)
            
            if "error" in verdict_data: # Check for error key in the returned dictionary
                     return JsonResponse({
                         "error": verdict_data['error'],
                         "status": "error"
                     }, status=500)

            return JsonResponse({
                'video_path': f"{settings.MEDIA_URL}output_videos/{final_output_name_encoded}",
                'result': verdict_data['verdict'],
                'report_path': verdict_data['report_path'],
                'audio': verdict_data['audio_verdict'],
                'lip_sync': verdict_data['lip_sync_verdict'],
                'visual': verdict_data['visual_verdict'],
                'status': 'success'
            })

        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)
'''


######################## deepfake V2 ####################

class AudioClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout_rate=0.3):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x).squeeze(-1)
        x = self.dropout2(x)
        return self.fc(x)

class LocalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure MTCNN is correctly initialized and available
        # It's better to pass the device from the main Deepfake class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.local_cnn = LocalCNN()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_lip_eye(self, face_img):
        h, w = face_img.shape[1], face_img.shape[2]
        lips = face_img[:, int(h*0.65):int(h*0.85), int(w*0.25):int(w*0.75)]
        eyes = face_img[:, int(h*0.2):int(h*0.45), int(w*0.2):int(w*0.8)]
        return lips.unsqueeze(0), eyes.unsqueeze(0)

    def forward(self, frame_np): # Expects a numpy array frame (BGR or RGB)
        # Convert numpy array to PIL Image for MTCNN
        frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)) # Assuming BGR input from OpenCV

        boxes, _ = self.mtcnn.detect(frame_pil)
        if boxes is None:
            return None # No face detected

        # Extract the largest face (MTCNN returns boxes in order of largest first if keep_all=False)
        x1, y1, x2, y2 = map(int, boxes[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_np.shape[1], x2), min(frame_np.shape[0], y2)

        face = frame_pil.crop((x1, y1, x2, y2))
        if face.size[0] == 0 or face.size[1] == 0:
            return None # Empty face crop

        face_tensor = self.transform(face).to(self.device) # Apply internal transform
        
        lips, eyes = self.extract_lip_eye(face_tensor) # lips/eyes are already tensors
        lips, eyes = lips.to(self.device), eyes.to(self.device)

        local_feat = torch.cat([
            self.local_cnn(lips),
            self.local_cnn(eyes)
        ], dim=1)
        
        global_feat = self.vit(face_tensor.unsqueeze(0)) # ViT expects batch dimension
        
        # Ensure global_feat is on the correct device. It should be from self.vit already
        return torch.cat([local_feat, global_feat], dim=1)

class DeepfakeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Assuming cnn_model is defined globally for SyncModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = timm.create_model('tf_efficientnet_b0', pretrained=True)
cnn_model.reset_classifier(0) # Resets classifier to 0 outputs, making it a feature extractor
cnn_model.eval().to(device)


class SyncModel(torch.nn.Module):
    def __init__(self, audio_feat_dim=768, visual_feat_dim=cnn_model.num_features):
        super().__init__()
        self.a_proj = torch.nn.Linear(audio_feat_dim, 128)
        self.v_proj = torch.nn.Linear(visual_feat_dim, 128)
        self.a_lstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.v_lstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.attn = torch.nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.classifier_bn_pool = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier_fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, audio, visual):
        a = self.a_proj(audio)
        v = self.v_proj(visual)
        a, _ = self.a_lstm(a)
        v, _ = self.v_lstm(v)
        fused, _ = self.attn(a, v, v)

        x = fused.transpose(1, 2)
        x = self.classifier_bn_pool(x).squeeze(-1)
        return torch.sigmoid(self.classifier_fc(x)).squeeze(-1)

# Ensure face_detector and facemark are loaded (from your original code)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
try:
    facemark_model_path = '/home/student/new_api/testpro-main/models/new_deepfake/lbfmodel.yaml'
    # Use resource_filename from pkg_resources if you need to load from an installed package
    # For local dev, direct path is fine.
    facemark.loadModel(facemark_model_path)
except Exception as e:
    print(f"Warning: Could not load facemark model at {facemark_model_path}. Lip sync check might be affected. Error: {e}")


class Deepfake_new(View):
    def __init__(self):
        super().__init__()
        self.colors = {
            'real': '#39FF14',  # Green
            'fake': '#FF0000'   # Red
        }
        self.text_color = '#000000'
        self.grid_color = '#AAAAAA'

        self.MODEL_PATHS = {
            'audio': '/home/student/new_api/testpro-main/models/new_deepfake/best_model_epoch_5.pth',
            'visual_extractor': "/home/student/new_api/testpro-main/models/new_deepfake/visual_feature_extractor.pth",
            'classifier': "/home/student/new_api/testpro-main/models/new_deepfake/deepfake_classifier.pth",
            'lbf':"/home/student/new_api/testpro-main/models/new_deepfake/lbfmodel.yaml",
            'sync_model': "/home/student/new_api/testpro-main/models/new_deepfake/sync_model_opencv_hubert_pytorch.pth",
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2.eval()
            self.wav2vec2.to(self.device)
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

            self.audio_model = AudioClassifier().to(self.device)
            state_dict_audio = torch.load(self.MODEL_PATHS['audio'], map_location=self.device)
            self.audio_model.load_state_dict(state_dict_audio, strict=False)
            self.audio_model.eval()

            self.extractor = VisualFeatureExtractor().to(self.device)
            state_dict_extractor = torch.load(self.MODEL_PATHS['visual_extractor'], map_location=self.device)
            self.extractor.load_state_dict(state_dict_extractor)
            self.extractor.eval()

            # Ensure the input_dim for DeepfakeClassifier is correct based on VisualFeatureExtractor's output
            # If VisualFeatureExtractor's output is 896, this is correct.
            self.model = DeepfakeClassifier(896).to(self.device)
            state_dict_classifier = torch.load(self.MODEL_PATHS['classifier'], map_location=self.device)
            self.model.load_state_dict(state_dict_classifier)
            self.model.eval()

            self.sync_model = SyncModel().to(self.device)
            state_dict_sync = torch.load(self.MODEL_PATHS['sync_model'], map_location=self.device)
            self.sync_model.load_state_dict(state_dict_sync)
            self.sync_model.eval()

            # NOTE: self.mtcnn here is used for initial face detection for display and cropping in predict_video
            # It's distinct from the MTCNN used inside VisualFeatureExtractor
            self.mtcnn = MTCNN(keep_all=False, device=self.device)

            # This transform was causing the original issue by being fed to DeepfakeClassifier directly
            # It's not directly used for the main DeepfakeClassifier now, as VisualFeatureExtractor handles its own transforms.
            # However, you might use it if you want to display the face crop at a specific size.
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)), # This size was for the classifier, now extractor takes care of it
                transforms.ToTensor(),
            ])


            self.LIP_LANDMARKS = list(range(48, 61))

            self.bundle = torchaudio.pipelines.HUBERT_BASE
            self.hubert_model = self.bundle.get_model().to(self.device).eval()
            self.hubert_sample_rate = self.bundle.sample_rate

            self.visual_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.segment_duration = 3

        except Exception as e:
            print(f"Error initializing Deepfake class: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Deepfake detector: {str(e)}")

    def _set_common_style(self):
        plt.rcParams.update({
            'axes.facecolor': (0, 0, 0, 0),
            'axes.edgecolor': self.grid_color,
            'axes.labelcolor': self.text_color,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': self.grid_color,
            'xtick.color': self.text_color,
            'ytick.color': self.text_color,
            'font.family': 'sans-serif',
            'font.weight': 'bold'
        })

    def calculate_scaled_dimensions(self, frame_width, frame_height, target_width, target_height):
        aspect_ratio = frame_width / frame_height
        target_aspect = target_width / target_height

        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            y_offset = (target_height - new_height) // 2
            return new_width, new_height, 0, y_offset
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            x_offset = (target_width - new_width) // 2
            return new_width, new_height, x_offset, 0

    def create_line_graph(self, prediction_history, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)
        self._set_common_style()

        frames = list(range(1, len(prediction_history) + 1))
        real_preds = [1 if p == 0 else 0 for p in prediction_history]
        fake_preds = [1 if p == 1 else 0 for p in prediction_history]

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.plot(frames, real_preds, label='Real', color=self.colors['real'], linewidth=1, alpha=0.8)
        ax.plot(frames, fake_preds, label='Fake', color=self.colors['fake'], linewidth=1, alpha=0.8)

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Real', 'Fake'])
        ax.set_title('Lip-sync Model Predictions Over Time', fontsize=6, fontweight='extra bold', color=self.text_color, pad=3)
        ax.set_xlabel('Segment Number', fontsize=6, color=self.text_color, fontweight='bold')
        ax.legend(fontsize=5, loc='upper right', facecolor='none', edgecolor='none')
        ax.tick_params(axis='both', labelsize=5, colors=self.text_color)

        plt.tight_layout(rect=[0, 0, 1, 1])
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return graph_img

    def create_pie_chart(self, real_count, fake_count, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)
        self._set_common_style()

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        labels = ['Real', 'Fake']
        sizes = [real_count, fake_count] if (real_count + fake_count) > 0 else [1, 1]
        colors = [self.colors['real'], self.colors['fake']]

        if real_count == 0 and fake_count == 0:
            autopct = lambda pct: 'No Data'
        else:
            autopct = '%1.1f%%'

        ax.pie(sizes, labels=labels, colors=colors, autopct=autopct, startangle=90,
            textprops={'fontsize': 5, 'color': self.text_color, 'fontweight': 'bold'}, radius=1.0)
        ax.set_title('Audio Model: Real vs Fake Ratio', fontsize=6, fontweight='extra bold', color=self.text_color, pad=3)

        plt.tight_layout(rect=[0, 0, 1, 1])
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return graph_img

    def create_bar_graph(self, real_count, fake_count, graph_width, graph_height):
        fig, ax = plt.subplots(figsize=(graph_width/130, graph_height/100), dpi=100)
        self._set_common_style()

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        labels = ['Real', 'Fake']
        counts = [real_count, fake_count]
        colors = [self.colors['real'], self.colors['fake']]

        ax.bar(labels, counts, color=colors, alpha=0.8)
        ax.set_ylim(0, max(5, real_count + fake_count + 5))
        ax.set_title('Visual Model: Real vs Fake Counts', fontsize=6, fontweight='extra bold', color=self.text_color, pad=3)
        ax.set_ylabel('Segment Count', fontsize=6, color=self.text_color, fontweight='bold')
        ax.tick_params(axis='both', labelsize=5, colors=self.text_color)
        ax.grid(True, alpha=0.15, color=self.grid_color)

        plt.tight_layout(rect=[0, 0, 1, 1])
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return graph_img

    def prepare_watermark(self, watermark_path, target_width, target_height):
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        if watermark is None:
            return None

        if watermark.shape[2] == 3:
            alpha = np.ones((watermark.shape[0], watermark.shape[1], 1), dtype=watermark.dtype) * 255
            watermark = cv2.merge([watermark, alpha])

        diagonal_length = int(np.sqrt(target_width*2 + target_height*2) * 0.4)
        aspect_ratio = watermark.shape[1] / watermark.shape[0]
        new_width = diagonal_length
        new_height = int(new_width / aspect_ratio)

        watermark_resized = cv2.resize(watermark, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        pad_size = max(new_width, new_height)
        padded = np.zeros((pad_size, pad_size, 4), dtype=np.uint8)
        x_offset = (pad_size - new_width) // 2
        y_offset = (pad_size - new_height) // 2

        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = watermark_resized

        angle = np.degrees(np.arctan2(target_height, target_width))
        matrix = cv2.getRotationMatrix2D((pad_size/2, pad_size/2), angle, 1.0)
        rotated = cv2.warpAffine(padded, matrix, (pad_size, pad_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        final_watermark = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        x_center_rotated = rotated.shape[1] // 2
        y_center_rotated = rotated.shape[0] // 2

        crop_x = x_center_rotated - target_width // 2
        crop_y = y_center_rotated - target_height // 2

        crop_x_start = max(0, crop_x)
        crop_y_start = max(0, crop_y)
        crop_x_end = min(rotated.shape[1], crop_x + target_width)
        crop_y_end = min(rotated.shape[0], crop_y + target_height)

        final_x_start = max(0, -crop_x)
        final_y_start = max(0, -crop_y)

        final_watermark[final_y_start : final_y_start + (crop_y_end - crop_y_start),
                                final_x_start : final_x_start + (crop_x_end - crop_x_start)] = rotated[
                                    crop_y_start:crop_y_end,
                                    crop_x_start:crop_x_end
                                ]
        return final_watermark

    def create_deepfake_graph_region(self, processed_frame_for_bg, graph_width, graph_height, bar_graph, pie_chart, line_graph, watermark_img=None):
        video_bg_source = cv2.resize(processed_frame_for_bg, (graph_width, graph_height))
        blurred_bg = cv2.GaussianBlur(video_bg_source, (15, 15), 10)

        gray_overlay = np.ones_like(blurred_bg) * 220
        blurred_overlay = cv2.GaussianBlur(gray_overlay, (49, 49), 15)

        blurred_bg = cv2.addWeighted(blurred_bg, 0.4, blurred_overlay, 0.6, 0)

        graph_region = blurred_bg.copy()

        padding = 15
        inner_graph_height = (graph_height - padding * 4) // 3
        inner_graph_width = graph_width - (padding * 2)

        inner_graph_height = max(1, inner_graph_height)
        inner_graph_width = max(1, inner_graph_width)

        y_start_bar = padding
        y_start_pie = padding + inner_graph_height + padding
        y_start_line = padding + inner_graph_height + padding + inner_graph_height + padding
        x_start = padding

        bar_graph_resized = cv2.resize(bar_graph, (inner_graph_width, inner_graph_height))
        if bar_graph_resized.shape[2] == 4:
            alpha_bar = bar_graph_resized[:, :, 3] / 255.0
            alpha_bar = np.dstack([alpha_bar] * 3)
            roi_bar = graph_region[y_start_bar:y_start_bar+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_bar.shape[:2] == bar_graph_resized.shape[:2]:
                graph_region[y_start_bar:y_start_bar+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_bar * (1 - alpha_bar) +
                    bar_graph_resized[:, :, :3] * alpha_bar
                ).astype(np.uint8)

        pie_chart_resized = cv2.resize(pie_chart, (inner_graph_width, inner_graph_height))
        if pie_chart_resized.shape[2] == 4:
            alpha_pie = pie_chart_resized[:, :, 3] / 255.0
            alpha_pie = np.dstack([alpha_pie] * 3)
            roi_pie = graph_region[y_start_pie:y_start_pie+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_pie.shape[:2] == pie_chart_resized.shape[:2]:
                graph_region[y_start_pie:y_start_pie+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_pie * (1 - alpha_pie) +
                    pie_chart_resized[:, :, :3] * alpha_pie
                ).astype(np.uint8)

        line_graph_resized = cv2.resize(line_graph, (inner_graph_width, inner_graph_height))
        if line_graph_resized.shape[2] == 4:
            alpha_line = line_graph_resized[:, :, 3] / 255.0
            alpha_line = np.dstack([alpha_line] * 3)
            roi_line = graph_region[y_start_line:y_start_line+inner_graph_height, x_start:x_start+inner_graph_width]
            if roi_line.shape[:2] == line_graph_resized.shape[:2]:
                graph_region[y_start_line:y_start_line+inner_graph_height, x_start:x_start+inner_graph_width] = (
                    roi_line * (1 - alpha_line) +
                    line_graph_resized[:, :, :3] * alpha_line
                ).astype(np.uint8)

        if watermark_img is not None:
            watermark_sized = cv2.resize(watermark_img, (graph_width, graph_height))
            if watermark_sized.shape[2] == 4:
                alpha_w = watermark_sized[:, :, 3] / 255.0
                alpha_w = np.dstack([alpha_w] * 3)
                graph_region = (graph_region * (1 - alpha_w * 0.4) +
                                watermark_sized[:, :, :3] * (alpha_w * 0.4)).astype(np.uint8)

        return graph_region

    def create_bordered_frame(self, main_content, heading_texts, output_width, output_height, heading_height, frame_width, real_count=0, fake_count=0):
        final_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        actual_titlebar_height = 40
        titlebar = np.ones((actual_titlebar_height, output_width, 3), dtype=np.uint8) * 255

        gradient = np.linspace(0.97, 1.0, actual_titlebar_height)[:, np.newaxis, np.newaxis]
        titlebar = (titlebar * gradient).astype(np.uint8)

        cv2.line(titlebar, (0, actual_titlebar_height-1), (output_width, actual_titlebar_height-1),
                 (240, 240, 240), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        text_color = (70, 70, 70)
        shadow_color = (200, 200, 200)
        text_thickness = 2

        temp_text_size = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_y = (actual_titlebar_height + temp_text_size[1]) // 2

        text_size_orig = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_x_orig = (frame_width - text_size_orig[0]) // 2

        cv2.putText(titlebar, heading_texts[0],
                            (text_x_orig+1, text_y+1),
                            font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[0],
                            (text_x_orig, text_y),
                            font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        text_size_foai = cv2.getTextSize(heading_texts[1], font, font_scale, text_thickness)[0]
        processed_panel_width = int(frame_width * 0.7)
        # right_panels_start_x = frame_width + processed_panel_width # This variable is not used
        # right_panels_width = output_width - right_panels_start_x # This variable is not used
        text_x_foai = frame_width + (processed_panel_width - text_size_foai[0]) // 2

        cv2.putText(titlebar, heading_texts[1],
                            (text_x_foai+1, text_y+1),
                            font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[1],
                            (text_x_foai, text_y),
                            font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        graph_panel_width = output_width - frame_width - processed_panel_width
        analysis_panel_start_x = frame_width + processed_panel_width

        font_size_metrics = 0.35
        text_thickness_metrics = 1

        title_padding = 5
        line_spacing = actual_titlebar_height / 3

        text_y1 = int(title_padding + line_spacing)
        text_y2 = int(title_padding + (line_spacing * 2))

        total_frames = max(1, real_count + fake_count)
        real_percent = (real_count / total_frames) * 100
        fake_percent = (fake_count / total_frames) * 100

        metric_area_width = graph_panel_width
        metric_padding = int(metric_area_width * 0.05)
        metric_x = analysis_panel_start_x + metric_padding

        metric_color = (255, 0, 0)

        cv2.putText(titlebar, f"Real: {real_percent:.1f}%",
                            (metric_x+1, text_y1+1),
                            font, font_size_metrics, shadow_color,
                            text_thickness_metrics, cv2.LINE_AA)
        cv2.putText(titlebar, f"Real: {real_percent:.1f}%",
                            (metric_x, text_y1),
                            font, font_size_metrics, metric_color,
                            text_thickness_metrics, cv2.LINE_AA)

        cv2.putText(titlebar, f"Fake: {fake_percent:.1f}%",
                            (metric_x+1, text_y2+1),
                            font, font_size_metrics, shadow_color,
                            text_thickness_metrics, cv2.LINE_AA)
        cv2.putText(titlebar, f"Fake: {fake_percent:.1f}%",
                            (metric_x, text_y2),
                            font, font_size_metrics, metric_color,
                            text_thickness_metrics, cv2.LINE_AA)

        final_frame[0:actual_titlebar_height, :] = titlebar
        content_y = actual_titlebar_height + 5
        content_height, content_width = main_content.shape[:2]
        final_frame[content_y:content_y+content_height, :content_width] = main_content

        return final_frame

    def extract_audio_ffmpeg(self, video_path, temp_audio_path):
        try:
            command = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-y', temp_audio_path
            ]
            process = subprocess.run(command, capture_output=True)
            if process.returncode != 0:
                print(f"FFmpeg audio extraction failed: {process.stderr.decode()}")
                return None
            return temp_audio_path
        except Exception as e:
            print(f"Error during FFmpeg audio extraction: {e}")
            return None

    def merge_video_audio_ffmpeg(self, video_path, audio_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            process = subprocess.run(command, capture_output=True)
            if process.returncode != 0:
                print(f"FFmpeg video-audio merge failed: {process.stderr.decode()}")
                # If merging fails, keep the video-only output and report
                if os.path.exists(video_path):
                    os.rename(video_path, output_path) # Rename original video to output_path
                return False
            return True
        except Exception as e:
            print(f"Error during FFmpeg video-audio merge: {e}")
            if os.path.exists(video_path):
                os.rename(video_path, output_path) # Rename original video to output_path
            return False

    def extract_audio_from_video_v1(self, video_path, audio_out_path="extracted_audio.wav"):
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_out_path, codec='pcm_s16le')
            video.audio.close()
            video.close()
            return audio_out_path
        except Exception as e:
            print(f"Error in extract_audio_from_video_v1: {e}")
            return None

    def split_audio(self, audio_path, chunk_duration=3, target_sr=16000):
        if not audio_path or not os.path.exists(audio_path):
            return [], target_sr
        try:
            wav, sr = librosa.load(audio_path, sr=target_sr)
            chunk_len = int(chunk_duration * target_sr)
            chunks = [wav[i:i+chunk_len] for i in range(0, len(wav), chunk_len) if len(wav[i:i+chunk_len]) == chunk_len]
            return chunks, target_sr
        except Exception as e:
            print(f"Error in split_audio: {e}")
            return [], target_sr

    def predict_chunks(self, chunks, processor, wav2vec2, model, device):
        model.eval()
        preds = []
        for chunk in chunks:
            try:
                inputs = processor(chunk, return_tensors="pt", sampling_rate=16000, padding=True).to(device)
                with torch.no_grad():
                    features = wav2vec2(inputs.input_values).last_hidden_state
                    outputs = model(features)
                    pred = outputs.argmax(dim=1).item()
                    preds.append(pred)
            except Exception as e:
                print(f"Error predicting audio chunk: {e}")
                preds.append(-1)
        return preds

    def classify_video(self, video_path, processor, wav2vec2, model, device):
        audio_path = self.extract_audio_from_video_v1(video_path)
        if not audio_path:
            return "Undetermined"
        chunks, sr = self.split_audio(audio_path)
        if not chunks:
            os.remove(audio_path)
            return "Undetermined"

        preds = self.predict_chunks(chunks, processor, wav2vec2, self.audio_model, device)
        os.remove(audio_path)

        valid_preds = [p for p in preds if p != -1]
        if not valid_preds:
            return "Undetermined"

        fake_count = valid_preds.count(1)
        fake_ratio = fake_count / len(valid_preds)

        label = "Fake" if fake_ratio >= 0.2 else "Real"
        return label

    def extract_audio_from_video_v2(self, video_path):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                audio_path = tmpfile.name
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=self.hubert_sample_rate, verbose=False, logger=None)
            clip.audio.close()
            clip.close()
            return audio_path
        except Exception as e:
            print(f"Error in extract_audio_from_video_v2: {e}")
            return None

    def split_segments(self, array, segment_len):
        if len(array) == 0 or segment_len <= 0:
            return []
        return [array[i:i+segment_len] for i in range(0, len(array), segment_len) if len(array[i:i+segment_len]) == segment_len]
    
    def extract_audio_features_video(self, video_path):
        try:
            audio_path = self.extract_audio_from_video_v2(video_path)
            if not audio_path:
                return []
            
            waveform, sr = torchaudio.load(audio_path)
            os.remove(audio_path)  # Clean up temp audio file
            
            if sr != self.hubert_sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.hubert_sample_rate)(waveform)

            with torch.no_grad():
                # Ensure waveform is 1-channel for Hubert
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                feats = self.hubert_model(waveform.to(self.device))[0][0].cpu().numpy()

            seg_len = int(self.hubert_sample_rate / 320) * self.segment_duration
            return self.split_segments(feats, seg_len)
            
        except Exception as e:
            print(f"Error extracting audio features for sync model: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_visual_features_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file for visual feature extraction (sync model).")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        lip_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Using face_detector from init for lip sync detection
                faces = face_detector.detectMultiScale(gray, 1.1, 5) 
                if not len(faces):
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if no face
                    continue
                face = faces[0] # Take the first detected face

                # Using facemark from init for lip sync detection
                ok, landmarks = facemark.fit(gray, np.array([face]))
                if not ok:
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if no landmarks
                    continue

                shape = landmarks[0][0]
                lip_pts = shape[self.LIP_LANDMARKS]
                x1, y1 = np.min(lip_pts, axis=0).astype(int)
                x2, y2 = np.max(lip_pts, axis=0).astype(int)
                pad = 5
                crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                                 max(0,x1-pad):min(frame.shape[1],x2+pad)]
                if crop.size == 0:
                    lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros if empty crop
                    continue

                tensor = self.visual_transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = cnn_model.forward_features(tensor)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1))
                    lip_features.append(pooled.view(-1).cpu().numpy())
            except Exception as e:
                print(f"Error processing visual frame for lip sync: {e}")
                import traceback
                traceback.print_exc()
                lip_features.append(np.zeros(cnn_model.num_features)) # Append zeros on error

        cap.release()
        seg_len = int(fps * self.segment_duration)
        return self.split_segments(np.array(lip_features), seg_len)


    def predict_sync(self, video_path, threshold=0.5):
        audio_segs = self.extract_audio_features_video(video_path)
        visual_segs = self.extract_visual_features_from_video(video_path)
        
        if not audio_segs or not visual_segs:
            print("Warning: Missing audio or visual segments for lip-sync prediction.")
            return "Undetermined", 0.0

        scores = []
        with torch.no_grad():
            min_len = min(len(audio_segs), len(visual_segs))
            for i in range(min_len):
                a_seg = audio_segs[i]
                v_seg = visual_segs[i]

                if len(a_seg) == 0 or len(v_seg) == 0:
                    print(f"Skipping sync segment {i} due to empty features.")
                    continue

                a_t = torch.tensor(a_seg, dtype=torch.float32).unsqueeze(0).to(self.device)
                v_t = torch.tensor(v_seg, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Check shapes before passing to sync_model.
                # a_t should be [1, sequence_length, audio_feature_dim (768)]
                # v_t should be [1, sequence_length, visual_feature_dim (cnn_model.num_features)]
                # self.sync_model expects [batch_size, seq_len, feature_dim]
                # If a_seg is [seq_len, 768] and v_seg is [seq_len, cnn_model.num_features]
                # Then unsqueeze(0) will make them [1, seq_len, feature_dim]
                
                if a_t.shape[1] > 0 and v_t.shape[1] > 0:
                    try:
                        scores.append(self.sync_model(a_t, v_t).item())
                    except Exception as e:
                        print(f"Error during sync model inference for segment {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Decide how to handle this error: append a default score, or skip?
                        # For now, it will just not append to scores, making it robust.
                else:
                    print(f"Skipping sync segment {i} due to zero sequence length after tensor conversion.")


        if not scores:
            print("No valid scores obtained for lip-sync prediction.")
            return "Undetermined", 0.0

        avg_score = float(np.mean(scores))
        label = "Fake" if avg_score > threshold else "Real"
        return label, avg_score

    def label_to_prob(self, label):
        return 1.0 if label == "Fake" else 0.0

    def predict_video(self, video_path, output_path):
        if self.model is None:
            return {"error": "Deepfake detection model not loaded."}

        report_data = [['Frame No.', 'Timestamp', 'Visual Verdict', 'Visual Confidence', 'Audio Verdict', 'Lip-Sync Verdict', 'Overall Weighted Verdict']]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open video file at {video_path}."}

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate panel widths based on original frame width
        processed_panel_width = int(fw * 0.7)
        graph_panel_width = int(fw * 0.3)
        # Ensure minimum sizes to avoid zero-dimension issues
        if fw < 100: # Arbitrary small width check
            processed_panel_width = max(100, processed_panel_width) # Ensure at least 100px
            graph_panel_width = max(50, graph_panel_width) # Ensure at least 50px

        output_width = fw + processed_panel_width + graph_panel_width
        output_width += output_width % 2 # Ensure even width for video writers

        heading_height = 45 if 45 % 2 == 0 else 46 # Ensure even height
        output_height = fh + heading_height
        output_height += output_height % 2 # Ensure even height

        graph_width = graph_panel_width # Corrected graph width
        graph_height = fh # Graphs will take up the full height of the video frame area

        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_audio_file_for_merge = os.path.join(temp_dir, f"temp_audio_merge_{timestamp_str}.wav")

        audio_path_for_merge = self.extract_audio_ffmpeg(video_path, temp_audio_file_for_merge)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard H.264 compatible codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        visual_real_count, visual_fake_count = 0, 0
        frame_count = 0
        visual_prediction_history = []
        last_box = None
        last_visual_prediction = -1 # Initialize to -1 (undetermined)

        watermark_path = os.path.join(settings.STATIC_ROOT, "testapp", "images","watermark.png")
        watermark = self.prepare_watermark(watermark_path, graph_width, graph_height)

        print("\n--------------------Audio Check---------------------\n")
        pred_audio = self.classify_video(video_path, self.processor, self.wav2vec2, self.audio_model, self.device)
        print(f"Audio-based prediction: {pred_audio}")

        print("\n--------------------Lip Sync Check---------------------\n")
        pred_sync, sync_score = self.predict_sync(video_path)
        print(f"Lip-sync based prediction: {pred_sync} (Score: {sync_score:.4f})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_frame = time.strftime('%H:%M:%S', time.gmtime(frame_count/fps))
            original_frame = frame.copy()
            annotated_frame = frame.copy() # Frame to draw detections on

            current_visual_prediction = -1 # Reset for each frame
            current_box = None
            visual_confidence_value = 0.0

            try:
                # Use VisualFeatureExtractor directly for classification
                # It handles its own MTCNN, cropping, resizing, and feature extraction
                visual_features = self.extractor.forward(original_frame) # extractor takes numpy frame (BGR assumed)

                if visual_features is not None:
                    # visual_features is already a tensor [1, 896]
                    with torch.no_grad():
                        output = self.model(visual_features) # DeepfakeClassifier expects features (896-dim)
                        prediction = torch.argmax(output, dim=1).item()
                        visual_confidence_value = torch.softmax(output, dim=1)[0][prediction].item() * 100
                        current_visual_prediction = prediction

                    # Get the box from the extractor's last detection for drawing
                    # This assumes extractor has a way to store/retrieve the last detected box
                    # For simplicity, let's re-run MTCNN on the original frame for bounding box display
                    # This avoids modifying VisualFeatureExtractor internal state unnecessarily for just display.
                    temp_img_for_box = Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                    box_for_display, conf_for_display = self.mtcnn.detect(temp_img_for_box)
                    if box_for_display is not None and conf_for_display is not None and conf_for_display[0] > 0.9:
                        x1, y1, x2, y2 = map(int, box_for_display[0])
                        current_box = (max(0, x1), max(0, y1), min(original_frame.shape[1], x2), min(original_frame.shape[0], y2))
                    else:
                        current_box = None # If MTCNN for display fails, no box for this frame


            except Exception as e:
                # IMPORTANT: Print the error to debug!
                print(f"Error during visual deepfake classification at frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                # If an error occurs, current_visual_prediction remains -1, which is handled
                # This leads to "Detecting..." or "No Face" on the display
                current_visual_prediction = -1 # Ensure it's explicitly -1 on error
                current_box = None # Clear box on error

            # Update last_box and last_visual_prediction for continuity
            if current_box is not None and current_visual_prediction != -1: # Only update if a valid prediction was made for this frame
                last_box = current_box
                last_visual_prediction = current_visual_prediction
            elif last_box is not None: # If no face or error this frame, but had a previous one, use it for display continuity
                current_box = last_box
                current_visual_prediction = last_visual_prediction # Use last valid prediction for display

            visual_prediction_history.append(current_visual_prediction)

            if current_visual_prediction == 0:
                visual_real_count += 1
            elif current_visual_prediction == 1:
                visual_fake_count += 1
            
            # Determine visual verdict for report based on the current frame's prediction
            visual_verdict_for_report = "Real" if current_visual_prediction == 0 else ("Fake" if current_visual_prediction == 1 else "No Face")
            
            visual_p_current = self.label_to_prob(visual_verdict_for_report)
            audio_p_overall = self.label_to_prob(pred_audio) # Overall audio verdict
            sync_p_overall = self.label_to_prob(pred_sync)   # Overall lip-sync verdict
            
            # Calculate overall weighted verdict for THIS FRAME (using overall audio/sync)
            combined_score_for_report_frame = visual_p_current * 0.3 + audio_p_overall * 0.2 + sync_p_overall * 0.5
            overall_weighted_verdict_for_report_frame = "Fake" if combined_score_for_report_frame >= 0.5 else "Real"

            report_data.append([
                str(frame_count),
                timestamp_frame,
                visual_verdict_for_report,
                f"{visual_confidence_value:.2f}%" if current_visual_prediction != -1 else "N/A",
                pred_audio, # Audio verdict is overall for the video
                pred_sync,  # Lip-sync verdict is overall for the video
                overall_weighted_verdict_for_report_frame
            ])

            # Drawing on the annotated_frame
            if current_box is not None:
                x1, y1, x2, y2 = current_box
                verdict_display_text = "Real" if current_visual_prediction == 0 else ("Fake" if current_visual_prediction == 1 else "No Face (Internal Error)") # Changed from "Detecting..."
                color = (0, 255, 0) if current_visual_prediction == 0 else ((0, 0, 255) if current_visual_prediction == 1 else (255, 255, 0)) # Yellow for "No Face"
                
                # Draw confidence if available
                confidence_text = f" Conf: {visual_confidence_value:.1f}%" if current_visual_prediction != -1 else ""

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f'Verdict: {verdict_display_text}{confidence_text}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                # This condition now correctly means no face was found or processed by VisualFeatureExtractor
                cv2.putText(annotated_frame, 'No Face Detected For Analysis', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA) # Orange for "No Face"

            # Resize annotated frame for processed panel display
            new_w_proc, new_h_proc, x_offset_proc, y_offset_proc = self.calculate_scaled_dimensions(
                fw, fh, processed_panel_width, fh
            )
            annotated_frame_resized = cv2.resize(annotated_frame, (new_w_proc, new_h_proc))
            processed_panel_frame = np.zeros((fh, processed_panel_width, 3), dtype=np.uint8)
            processed_panel_frame[y_offset_proc : y_offset_proc + new_h_proc,
                                        x_offset_proc : x_offset_proc + new_w_proc] = annotated_frame_resized

            # Graph generation
            graph_height_per_chart = graph_height // 3
            bar_graph_img = self.create_bar_graph(visual_real_count, visual_fake_count, graph_width, graph_height_per_chart)
            pie_chart_img = self.create_pie_chart(visual_real_count, visual_fake_count, graph_width, graph_height_per_chart)
            line_graph_img = self.create_line_graph(visual_prediction_history, graph_width, graph_height_per_chart)

            graph_region = self.create_deepfake_graph_region(
                processed_panel_frame, # Use processed frame for background
                graph_width,
                graph_height,
                bar_graph_img,
                pie_chart_img,
                line_graph_img,
                watermark
            )

            # Concatenate panels horizontally
            main_content = cv2.hconcat([original_frame, processed_panel_frame, graph_region])

            # Create final frame with title bar
            final_frame = self.create_bordered_frame(
                main_content,
                ["Original Video", "FOAI"],
                output_width,
                output_height,
                heading_height,
                fw,
                real_count=visual_real_count,
                fake_count=visual_fake_count
            )

            out.write(final_frame)

        cap.release()
        out.release()

        # Final verdicts for the summary report
        visual_p_final = self.label_to_prob("Fake" if visual_fake_count > visual_real_count else "Real")
        audio_p_final = self.label_to_prob(pred_audio)
        sync_p_final = self.label_to_prob(pred_sync)
        
        combined_score_final = visual_p_final * 0.3 + audio_p_final * 0.2 + sync_p_final * 0.5
        final_verdict_summary = "Fake" if combined_score_final >= 0.5 else "Real"

        if visual_real_count == 0 and visual_fake_count == 0:
            final_verdict_summary = "Undetermined (No faces detected for visual analysis)"

        report_path = os.path.join(settings.MEDIA_ROOT, f"reports/deepfake_report_{timestamp_str}.pdf")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Deepfake Detection Report", styles['Heading1']))
        elements.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Overall Weighted Verdict: <b>{final_verdict_summary}</b>", styles['Normal']))
        elements.append(Paragraph(f"Visual (Frame-based) Analysis: Real Frames: {visual_real_count}, Fake Frames: {visual_fake_count}", styles['Normal']))
        elements.append(Paragraph(f"Audio-based Prediction: {pred_audio}", styles['Normal']))
        elements.append(Paragraph(f"Lip-Sync Prediction: {pred_sync} (Average Score: {sync_score:.4f})", styles['Normal']))
        elements.append(Paragraph("<br/><br/>", styles['Normal']))
        
        table = Table(report_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        
        doc.build(elements)

        if audio_path_for_merge and os.path.exists(audio_path_for_merge):
            temp_video_with_audio_merged = os.path.join(temp_dir, f"temp_merged_video_{timestamp_str}.mp4")
            merge_success = self.merge_video_audio_ffmpeg(output_path, audio_path_for_merge, temp_video_with_audio_merged)
            if merge_success:
                os.remove(output_path)
                os.rename(temp_video_with_audio_merged, output_path)
            else:
                print(f"Failed to merge audio. Video saved without audio at {output_path}")
        else:
            print(f"No audio extracted or audio file not found. Video saved without audio at {output_path}")

        try:
            if audio_path_for_merge and os.path.exists(audio_path_for_merge):
                os.remove(audio_path_for_merge)
        except Exception as e:
            print(f"Error cleaning up temporary audio file: {e}")
            pass

        return {
            'verdict': final_verdict_summary,
            'report_path': f"{settings.MEDIA_URL}reports/deepfake_report_{timestamp_str}.pdf",
            'audio_verdict': pred_audio,
            'lip_sync_verdict': pred_sync,
            'visual_verdict': "Fake" if visual_fake_count > visual_real_count else "Real" if (visual_real_count + visual_fake_count) > 0 else "Undetermined"
        }

    def encode_video(self, input_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding failed: {process.stderr.decode()}")
                return None

            return output_path

        except Exception as e:
            print(f"Error during video encoding: {e}")
            return None
   
    def post(self, request):
        if self.model is None:
            return JsonResponse({"error": "Deepfake detection model failed to load."}, status=500)

        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)
            
            video_uuid = request.session.get("video_uuid")
            if not video_uuid:
                return JsonResponse({"error": "No video UUID found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            output_dir = os.path.join(settings.MEDIA_ROOT, "output_videos")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_output_name = f"temp_deepfake_analysis_{video_uuid}.mp4"
            final_output_name = f"deepfake_analysis_{video_uuid}.mp4"

            temp_processed_path = os.path.join(output_dir, temp_output_name)
            final_output_path = os.path.join(output_dir, final_output_name)

            verdict_data = self.predict_video(video_absolute_path, temp_processed_path)

            if "error" in verdict_data: # Check for error key in the returned dictionary
                     return JsonResponse({
                         "error": verdict_data['error'],
                         "status": "error"
                     }, status=500)

            # Encode the final output video
            encoded_video_path = self.encode_video(temp_processed_path, final_output_path)
            if not encoded_video_path:
                return JsonResponse({
                    "error": "Failed to encode the final output video.",
                    "status": "error"
                }, status=500)
            # If encoding was successful, return the final output path
            


            return JsonResponse({
                'video_path': f"{settings.MEDIA_URL}output_videos/{final_output_name}",
                #'result': verdict_data['verdict'],
                'report_path': verdict_data['report_path'],
                'audio': verdict_data['audio_verdict'],
                'lip_sync': verdict_data['lip_sync_verdict'],
                'visual': verdict_data['visual_verdict'],
                'status': 'success'
            })

        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)



################################ Posture Detection class ##############################

class PostureV2(View):
    # Check GPU availability for TensorFlow
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("✅ GPU detected and configured for TensorFlow")
        except Exception as e:
            print(f"⚠️ GPU configuration error: {e}. Falling back to CPU.")
    else:
        print("❌ No GPU detected for TensorFlow. Using CPU.")



    def __init__(self):
        # ---------- LABELS & GLOBALS ----------
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.log_data = []
        self.pose_vector_log = []
        self.lstm_input_log = []
        self.OPTIMIZE_EVERY_N_FRAMES = 3  # Optimize every 3rd frame
        self.last_posture_score = 50  # Cache posture score
        self.SMOOTHING_WINDOW = 10  # Frames for emotion smoothing (~0.33s at 30 FPS)
        self.SMOOTHING_ALPHA = 0.3  # EMA smoothing factor
        self.LSTM_SEQUENCE_LENGTH = 10  # Frames per LSTM sequence
        self.emotion_probs_buffer = deque(maxlen=self.SMOOTHING_WINDOW)  # Buffer for emotion probabilities

        # ---------- Mediapipe Setup (Pose Only) ----------
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # ---------- Load Emotion Detection Model ----------
        try:
            self.emotion_model = tf.keras.models.load_model("/home/student/new_api/testpro-main/models/posture/Ar-emotiondetector.h5")
            print("✅ Emotion model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            sys.exit()

        # ---------- Load YOLOv10 Face Detection Model ----------
        try:
            self.yolo_model = YOLO("/home/student/new_api/testpro-main/models/posture/yolov10n-face.pt")  # Load pre-trained face model
            print("✅ YOLO face detection model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            sys.exit()

    def preprocess_face(self,face_img):
        """Preprocess face image for emotion detection"""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = tf.keras.preprocessing.image.img_to_array(face_img) / 255.0
        return np.expand_dims(face_img, axis=0)

    def calculate_angle(self,a, b, c):
        """Calculate angle between three points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def draw_text(self,img, text, pos, color=(0, 255, 0), size=0.7, thickness=2):
        """Draw text on image"""
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    def draw_warning(self,img, text, pos):
        """Draw warning text on image"""
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def extract_pose_vector(self,landmarks, width, height):
        """
        Extracts 66D [x1, y1, ..., x33, y33] pose vector scaled to pixel space.
        """
        vec = []
        for lm in landmarks:
            vec.append(lm.x * width)
            vec.append(lm.y * height)
        return vec

    def smooth_emotion_probs(self,new_probs, buffer):
        """
        Apply exponential moving average to emotion probabilities.
        """
        alpha=self.SMOOTHING_ALPHA
        buffer.append(new_probs)
        if len(buffer) < 2:
            return new_probs
        smoothed = np.zeros_like(new_probs)
        for probs in buffer:
            smoothed = alpha * probs + (1 - alpha) * smoothed
        return smoothed / np.sum(smoothed)  # Normalize

    def predict_emotion(self,face_img):
        """Predict emotion from face image"""
        if face_img is not None and face_img.size != 0:
            try:
                face_input = self.preprocess_face(face_img)
                preds = self.emotion_model.predict(face_input, verbose=0)[0]
                
                # Ensure we have 7 predictions for 7 emotions
                if len(preds) != 7:
                    print(f"⚠️ Model output size mismatch. Expected 7, got {len(preds)}")
                    # Create default neutral-biased distribution
                    preds = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
                
                # Normalize probabilities
                preds = preds / np.sum(preds)
                face_emotion = self.emotion_labels[np.argmax(preds)]
                return preds, face_emotion
            except Exception as e:
                print(f"Emotion prediction error: {e}")
                # Return neutral as default with neutral-biased probabilities
                return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]), "Neutral"
        else:
            # No face detected - return neutral-biased distribution
            return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]), "Neutral"

    def detect_neutral_state(self,emotion_probs, posture_score, face_confidence, movement_score=0):
        """
        Enhanced neutral detection considering multiple factors
        """
        neutral_indicators = 0
        
        # Check if emotion probabilities are relatively uniform (indicating uncertainty)
        emotion_entropy = -np.sum(emotion_probs * np.log(emotion_probs + 1e-8))
        if emotion_entropy > 1.5:  # High entropy indicates uncertainty
            neutral_indicators += 1
        
        # Check if posture score is in neutral range
        if 40 <= posture_score <= 70:
            neutral_indicators += 1
        
        # Check if face confidence is low (might indicate neutral expression)
        if face_confidence < 0.5:
            neutral_indicators += 1
        
        # If multiple indicators suggest neutral state, boost neutral probability
        if neutral_indicators >= 2:
            boosted_probs = emotion_probs.copy()
            boosted_probs[6] *= 1.3  # Boost neutral probability (index 6 for Neutral)
            boosted_probs = boosted_probs / np.sum(boosted_probs)
            return boosted_probs
        
        return emotion_probs

    def fuse_posture_emotion(self,emotion_probs, posture_score, face_confidence):
        """Fuse emotion probabilities with posture information"""
        # Dynamic weighting based on face confidence
        if face_confidence > 0.7:
            w_face, w_posture = 0.7, 0.3  # Trust face more when confidence is high
        elif face_confidence > 0.4:
            w_face, w_posture = 0.5, 0.5  # Balanced when moderate confidence
        else:
            w_face, w_posture = 0.3, 0.7  # Trust posture more when face confidence is low
        
        # Create posture emotion vector (7D for 7 emotions)
        # [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
        posture_vec = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])  # Default neutral
        
        if posture_score > 75:
            # Excellent posture - more likely to be Happy or Neutral
            posture_vec = np.array([0.05, 0.05, 0.05, 0.35, 0.05, 0.15, 0.3])
        elif posture_score > 60:
            # Good posture - likely Happy or Neutral
            posture_vec = np.array([0.08, 0.08, 0.07, 0.25, 0.07, 0.15, 0.3])
        elif posture_score > 40:
            # Average posture - likely Neutral
            posture_vec = np.array([0.12, 0.1, 0.1, 0.15, 0.13, 0.1, 0.3])
        elif posture_score > 25:
            # Poor posture - might be Sad, Angry, or Neutral
            posture_vec = np.array([0.2, 0.1, 0.15, 0.1, 0.25, 0.05, 0.15])
        else:
            # Very poor posture - likely Sad or Angry
            posture_vec = np.array([0.3, 0.1, 0.2, 0.05, 0.3, 0.03, 0.02])
        
        # Ensure probabilities sum to 1
        posture_vec = posture_vec / np.sum(posture_vec)
        emotion_probs = emotion_probs / np.sum(emotion_probs)
        
        # Fuse the probabilities
        fused = w_face * emotion_probs + w_posture * posture_vec
        fused = fused / np.sum(fused)  # Normalize
        
        return fused, self.emotion_labels[np.argmax(fused)]

    # ---------- OPTIMIZERS ----------
    def grey_wolf_optimizer(self,fitness_func, bounds, num_agents=3, max_iter=3, angles=None):
        dim = len(bounds)
        alpha = np.zeros(dim)
        alpha_score = float("inf")
        wolves = np.random.rand(num_agents, dim)
        for i in range(dim):
            wolves[:, i] = wolves[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        for t in range(max_iter):
            for i in range(num_agents):
                fitness = fitness_func(wolves[i], angles)
                if fitness < alpha_score:
                    alpha_score, alpha = fitness, wolves[i].copy()
            a = 2 - t * (2 / max_iter)
            for i in range(num_agents):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    wolves[i][j] = alpha[j] - A1 * D_alpha
        return alpha

    def grasshopper_optimizer(self,fitness_func, bounds, num_agents=3, max_iter=3, angles=None):
        dim = len(bounds)
        c_max, c_min = 1.0, 0.00004
        positions = np.random.rand(num_agents, dim)
        for i in range(dim):
            positions[:, i] = positions[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        fitness = np.array([fitness_func(pos, angles) for pos in positions])
        best_pos = positions[np.argmin(fitness)].copy()
        for t in range(max_iter):
            c = c_max - t * ((c_max - c_min) / max_iter)
            for i in range(num_agents):
                positions[i] = c * np.random.rand(dim) + best_pos
                for d in range(dim):
                    positions[i, d] = np.clip(positions[i, d], bounds[d][0], bounds[d][1])
            fitness = np.array([fitness_func(pos, angles) for pos in positions])
            best_pos = positions[np.argmin(fitness)].copy()
        return best_pos

    def particle_swarm_optimizer(self,fitness_func, bounds, num_particles=3, max_iter=3, angles=None):
        dim = len(bounds)
        particles = np.random.rand(num_particles, dim)
        velocities = np.zeros_like(particles)
        for i in range(dim):
            particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        p_best = particles.copy()
        p_best_scores = np.array([fitness_func(p, angles) for p in particles])
        g_best = p_best[np.argmin(p_best_scores)]
        for t in range(max_iter):
            for i in range(num_particles):
                fitness = fitness_func(particles[i], angles)
                if fitness < p_best_scores[i]:
                    p_best_scores[i], p_best[i] = fitness, particles[i].copy()
            g_best = p_best[np.argmin(p_best_scores)]
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = 0.5 * velocities[i] + 0.8 * r1 * (p_best[i] - particles[i]) + 0.9 * r2 * (g_best - particles[i])
                particles[i] += velocities[i]
        return g_best

    def fitness_function(self,params, angles):
        """Fitness function for posture optimization"""
        ideal = np.array([90, 60, 45, 120])  # Back, neck, shoulder, elbow
        weights = [1.0, 1.2, 0.8, 1.0]
        error = np.sum(((np.array(angles) - ideal) ** 2) * weights)
        posture_score = params[0]
        return error + abs(posture_score - 50)

    def optimize_posture_score(self,angles, frame_num):
        """Optimize posture score using hybrid optimization"""
        self.last_posture_score
        if frame_num % self.OPTIMIZE_EVERY_N_FRAMES == 0:
            bounds = [(0, 100)]
            try:
                gwo = self.grey_wolf_optimizer(self.fitness_function, bounds, angles=angles)[0]
                goa = self.grasshopper_optimizer(self.fitness_function, bounds, angles=angles)[0]
                pso = self.particle_swarm_optimizer(self.fitness_function, bounds, angles=angles)[0]
                self.last_posture_score = (gwo + goa + pso) / 3
            except Exception as e:
                print(f"Optimization error: {e}")
                # Fallback to simple calculation
                ideal_angles = [90, 60, 45, 120]
                deviations = [abs(angles[i] - ideal_angles[i]) for i in range(len(angles))]
                avg_deviation = np.mean(deviations)
                self.last_posture_score = max(0, 100 - avg_deviation)
        
        return self.last_posture_score

    def process_frame_emotions(self,face_img, posture_score, face_confidence):
        """Process emotions for a single frame"""
        
        # Predict face emotion
        emotion_probs, face_emotion = self.predict_emotion(face_img)
        
        # Apply neutral detection enhancement
        enhanced_probs = self.detect_neutral_state(emotion_probs, posture_score, face_confidence)
        
        # Fuse with posture information
        fused_probs, fused_emotion = self.fuse_posture_emotion(enhanced_probs, posture_score, face_confidence)
        
        # Apply temporal smoothing
        smoothed_probs = self.smooth_emotion_probs(fused_probs, self.emotion_probs_buffer)
        smoothed_emotion = self.emotion_labels[np.argmax(smoothed_probs)]
        
        return {
            'face_emotion': face_emotion,
            'face_probs': emotion_probs,
            'enhanced_probs': enhanced_probs,
            'fused_emotion': fused_emotion,
            'fused_probs': fused_probs,
            'smoothed_emotion': smoothed_emotion,
            'smoothed_probs': smoothed_probs
        }

    # ---------- MAIN ----------
    def main_analysis(self,input_video,output_video):
        optimizer_type = 'HYBRID'

        if optimizer_type != "HYBRID":
            print("❌ Only HYBRID is supported for fusion in this version.")
            sys.exit()

        print("🎬 Select a video file to proceed...")
        video_path = input_video
        if not video_path:
            print("❌ No video selected.")
            sys.exit()

        print("⚙️ Using HYBRID (GWO + GOA + PSO) for posture score with enhanced neutral detection")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error opening video file")
            sys.exit()
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        temp_output = output_video.replace(".mp4", "_temp.mp4")
        out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        lstm_sequence_buffer = deque(maxlen=self.LSTM_SEQUENCE_LENGTH)  # Buffer for LSTM sequences
        sequence_id = 0

        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Progress indicator
                if frame_num % 100 == 0:
                    print(f"Processing frame {frame_num}/{total_frames} ({(frame_num/total_frames)*100:.1f}%)")

                scale = 0.5
                resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = True

                # Pose detection
                pose_result = pose.process(img_rgb)
                back_angle = neck_angle = shoulder_angle = elbow_angle = 0
                posture_score = 50
                pose_vec = [0] * 66  # Default pose vector

                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    try:
                        # Back angle: shoulder-hip-knee
                        shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                                    lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                        hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                        knee = [lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                        back_angle = self.calculate_angle(shoulder, hip, knee)

                        # Neck angle: ear-shoulder-hip
                        ear = [lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * width,
                            lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * height]
                        neck_angle = self.calculate_angle(ear, shoulder, hip)

                        # Shoulder angle: elbow-shoulder-hip
                        elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                        shoulder_angle = self.calculate_angle(elbow, shoulder, hip)

                        # Elbow angle: wrist-elbow-shoulder
                        wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                        elbow_angle = self.calculate_angle(wrist, elbow, shoulder)

                        angles = [back_angle, neck_angle, shoulder_angle, elbow_angle]
                        posture_score = self.optimize_posture_score(angles, frame_num)

                        # Extract 66D pose vector
                        pose_vec = self.extract_pose_vector(lm, width, height)
                        pose_entry = {"frame": frame_num}
                        for i, val in enumerate(pose_vec):
                            pose_entry[f"pose_{i}"] = round(val, 2)
                        self.pose_vector_log.append(pose_entry)
                    except Exception as e:
                        print(f"Angle calculation error: {e}")

                    self.mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # YOLOv10 Face Detection
                face_img = None
                face_confidence = 0.0  # Default for no detection
                try:
                    results = self.yolo_model(resized, conf=0.25, iou=0.7, imgsz=640, verbose=False)
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        if len(boxes) > 0:
                            x1, y1, x2, y2 = boxes[0]  # First face
                            face_confidence = confidences[0]
                            x1, y1, x2, y2 = [int(coord / scale) for coord in [x1, y1, x2, y2]]
                            x, y = max(0, x1), max(0, y1)
                            w, h = min(x2 - x1, width - x), min(y2 - y1, height - y)
                            if w > 0 and h > 0:
                                face_img = frame[y:y + h, x:x + w]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            break
                except Exception as e:
                    print(f"Face detection error: {e}")

                # Process emotions
                emotion_results = self.process_frame_emotions(face_img, posture_score, face_confidence)

                # Prepare LSTM input sequence
                sequence_entry = {
                    "frame": frame_num,
                    "sequence_id": sequence_id,
                    **{f"emotion_prob_{i}": round(p, 4) for i, p in enumerate(emotion_results['fused_probs'])},
                    **{f"pose_vec_{i}": round(v, 2) for i, v in enumerate(pose_vec)}
                }
                lstm_sequence_buffer.append(sequence_entry)
                if len(lstm_sequence_buffer) == self.LSTM_SEQUENCE_LENGTH:
                    for i, entry in enumerate(lstm_sequence_buffer):
                        self.lstm_input_log.append({
                            "sequence_id": sequence_id,
                            "frame": entry["frame"],
                            "timestep": i,
                            **{k: v for k, v in entry.items() if k not in ["frame", "sequence_id"]}
                        })
                    sequence_id += 1

                # Draw information on frame
                self.draw_text(frame, f"Back Angle: {int(back_angle)}", (30, 30))
                self.draw_text(frame, f"Neck Angle: {int(neck_angle)}", (30, 60))
                self.draw_text(frame, f"Shoulder Angle: {int(shoulder_angle)}", (30, 90))
                self.draw_text(frame, f"Elbow Angle: {int(elbow_angle)}", (30, 120))
                self.draw_text(frame, f"Posture Score: {int(posture_score)}", (30, 150))
                
                # Posture warnings
                if back_angle < 60 or back_angle > 120:
                    self.draw_warning(frame, "Bad Back Posture!", (30, 180))
                if neck_angle < 40 or neck_angle > 90:
                    self.draw_warning(frame, "Bad Neck Posture!", (30, 210))

                # Emotion information
                self.draw_text(frame, f"Face Emotion: {emotion_results['face_emotion']}", (width - 300, 30), (255, 255, 255))
                self.draw_text(frame, f"Fused Emotion: {emotion_results['fused_emotion']}", (width - 300, 60), (0, 255, 255))
                self.draw_text(frame, f"Final Emotion: {emotion_results['smoothed_emotion']}", (width - 300, 90), (255, 255, 0))
                self.draw_text(frame, f"Face Confidence: {face_confidence:.2f}", (width - 300, 120), (255, 255, 0))
                
                # Emotion probabilities (top 3)
                sorted_indices = np.argsort(emotion_results['smoothed_probs'])[::-1][:3]
                for i, idx in enumerate(sorted_indices):
                    prob = emotion_results['smoothed_probs'][idx]
                    emotion = self.emotion_labels[idx]
                    self.draw_text(frame, f"{emotion}: {prob:.2f}", (width - 300, 150 + i*30), (200, 200, 200), size=0.5)

                # Log data
                self.log_data.append({
                    "frame": frame_num,
                    "back_angle": round(back_angle, 2),
                    "neck_angle": round(neck_angle, 2),
                    "shoulder_angle": round(shoulder_angle, 2),
                    "elbow_angle": round(elbow_angle, 2),
                    "posture_score": round(posture_score, 2),
                    "face_emotion": emotion_results['face_emotion'],
                    "fused_emotion": emotion_results['fused_emotion'],
                    "final_emotion": emotion_results['smoothed_emotion'],
                    "face_confidence": round(face_confidence, 3),
                    **{f"prob_{self.emotion_labels[i]}": round(emotion_results['smoothed_probs'][i], 3) for i in range(7)}
                })

                out.write(frame)
                frame_num += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_video
            ]

            subprocess.run(ffmpeg_cmd, check=True)
            os.remove(temp_output)  # Remove temporary file after conversion
        except Exception as e:
            print("❌ Error during video processing:", e)

        # Save results
        print("\n💾 Saving results...")
        
        df = pd.DataFrame(self.log_data)
        df.to_csv("emotion_posture_log.csv", index=False)
        print("✅ Saved emotion and posture log: emotion_posture_log.csv")

        if self.pose_vector_log:
            pose_df = pd.DataFrame(self.pose_vector_log)
            pose_df.to_csv("pose_vector_log.csv", index=False)
            print("✅ Saved 66D pose vectors to: pose_vector_log.csv")

        if self.lstm_input_log:
            lstm_df = pd.DataFrame(self.lstm_input_log)
            lstm_df.to_csv("lstm_input_log.csv", index=False)
            print("✅ Saved LSTM input sequences to: lstm_input_log.csv")



        Emo_des = []
        # Final emotion analysis
        if self.log_data:
            final_emotions = [entry['final_emotion'] for entry in self.log_data]
            emotion_counts = Counter(final_emotions)
            

            

            print("\n📊 Emotion Distribution:")
            for emotion, count in emotion_counts.most_common():
                percentage = (count / len(final_emotions)) * 100
                print(f"  {emotion}: {count} frames ({percentage:.1f}%)")
                Emo_des.append(f"{emotion}: {count} frames ({percentage:.1f}%)")
            
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            print(f"\n🎯 Dominant Emotion: {dominant_emotion}")
            
            # Neutral detection statistics
            neutral_percentage = (emotion_counts.get('Neutral', 0) / len(final_emotions)) * 100
            print(f"🔍 Neutral Detection: {neutral_percentage:.1f}% of frames")
            
        else:
            print("❌ No frames processed")

        print(f"\n🎬 Processed video saved as: {output_video}")

        return Emo_des,dominant_emotion

    def post(self, request):
        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)
            
            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)
            
            video_uuid = request.session.get("video_uuid")
            if not video_uuid:
                return JsonResponse({"error": "No video UUID found in session"}, status=400)

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"posture_video_{video_uuid}.mp4"

            output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, output_filename)
            

            emotion_distribution,dominant_emotion = self.main_analysis(video_absolute_path, output_path)


            return JsonResponse({
                'status': 'success',
                'video_path': f"{settings.MEDIA_URL}output/{output_filename}",
                'emotion_distribution':emotion_distribution,
                'dominant_emotion':dominant_emotion,
            })
        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)

'''
class Posture(View):

    def __init__(self):
        super().__init__()
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("✅ GPU detected and configured for TensorFlow")
            except Exception as e:
                print(f"⚠️ GPU configuration error: {e}. Falling back to CPU.")
        else:
            print("❌ No GPU detected for TensorFlow. Using CPU.")

    # ---------- LABELS & GLOBALS ----------
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'No Emotion']
        self.log_data = []
        self.pose_vector_log = []
        self.lstm_input_log = []
        self.OPTIMIZE_EVERY_N_FRAMES = 3  # Optimize every 3rd frame
        self.last_posture_score = 50  # Cache posture score
        self.SMOOTHING_WINDOW = 10  # Frames for emotion smoothing (~0.33s at 30 FPS)
        self.SMOOTHING_ALPHA = 0.3  # EMA smoothing factor
        self.LSTM_SEQUENCE_LENGTH = 10  # Frames per LSTM sequence
        self.emotion_probs_buffer = deque(maxlen=self.SMOOTHING_WINDOW)  # Buffer for emotion probabilities

    # ---------- Mediapipe Setup (Pose Only) ----------
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # ---------- Load Emotion Detection Model ----------
        self.emotion_model = tf.keras.models.load_model("/home/student/new_api/testpro-main/models/Ar-emotiondetector.h5")

        # ---------- Load YOLOv10 Face Detection Model ----------
        self.yolo_model = YOLO("/home/student/new_api/testpro-main/models/yolov10n-face.pt")  # Load pre-trained face model

    def preprocess_face(self,face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = tf.keras.preprocessing.image.img_to_array(face_img) / 255.0
        return np.expand_dims(face_img, axis=0)

    def calculate_angle(self,a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def draw_text(self,img, text, pos, color=(0, 255, 0), size=0.7, thickness=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    def draw_warning(self,img, text, pos):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def extract_pose_vector(self,landmarks, width, height):
        """
        Extracts 66D [x1, y1, ..., x33, y33] pose vector scaled to pixel space.
        """
        vec = []
        for lm in landmarks:
            vec.append(lm.x * width)
            vec.append(lm.y * height)
        return vec

    def smooth_emotion_probs(self,new_probs, buffer):
        """
        Apply exponential moving average to emotion probabilities.
        """
        alpha=self.SMOOTHING_ALPHA
        buffer.append(new_probs)
        if len(buffer) < 2:
            return new_probs
        smoothed = np.zeros_like(new_probs)
        for probs in buffer:
            smoothed = alpha * probs + (1 - alpha) * smoothed
        return smoothed

    # ---------- OPTIMIZERS ----------
    def grey_wolf_optimizer(self,fitness_func, bounds, num_agents=3, max_iter=3, angles=None):
        dim = len(bounds)
        alpha = np.zeros(dim)
        alpha_score = float("inf")
        wolves = np.random.rand(num_agents, dim)
        for i in range(dim):
            wolves[:, i] = wolves[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        for t in range(max_iter):
            for i in range(num_agents):
                fitness = fitness_func(wolves[i], angles)
                if fitness < alpha_score:
                    alpha_score, alpha = fitness, wolves[i].copy()
            a = 2 - t * (2 / max_iter)
            for i in range(num_agents):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    wolves[i][j] = alpha[j] - A1 * D_alpha
        return alpha

    def grasshopper_optimizer(self,fitness_func, bounds, num_agents=3, max_iter=3, angles=None):
        dim = len(bounds)
        c_max, c_min = 1.0, 0.00004
        positions = np.random.rand(num_agents, dim)
        for i in range(dim):
            positions[:, i] = positions[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        fitness = np.array([fitness_func(pos, angles) for pos in positions])
        best_pos = positions[np.argmin(fitness)].copy()
        for t in range(max_iter):
            c = c_max - t * ((c_max - c_min) / max_iter)
            for i in range(num_agents):
                positions[i] = c * np.random.rand(dim) + best_pos
                for d in range(dim):
                    positions[i, d] = np.clip(positions[i, d], bounds[d][0], bounds[d][1])
            fitness = np.array([fitness_func(pos, angles) for pos in positions])
            best_pos = positions[np.argmin(fitness)].copy()
        return best_pos

    def particle_swarm_optimizer(self,fitness_func, bounds, num_particles=3, max_iter=3, angles=None):
        dim = len(bounds)
        particles = np.random.rand(num_particles, dim)
        velocities = np.zeros_like(particles)
        for i in range(dim):
            particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        p_best = particles.copy()
        p_best_scores = np.array([fitness_func(p, angles) for p in particles])
        g_best = p_best[np.argmin(p_best_scores)]
        for t in range(max_iter):
            for i in range(num_particles):
                fitness = fitness_func(particles[i], angles)
                if fitness < p_best_scores[i]:
                    p_best_scores[i], p_best[i] = fitness, particles[i].copy()
            g_best = p_best[np.argmin(p_best_scores)]
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = 0.5 * velocities[i] + 0.8 * r1 * (p_best[i] - particles[i]) + 0.9 * r2 * (g_best - particles[i])
                particles[i] += velocities[i]
        return g_best

    def fitness_function(self,params, angles):
        ideal = np.array([90, 60, 45, 120])  # Back, neck, shoulder, elbow
        weights = [1.0, 1.2, 0.8, 1.0]
        error = np.sum(((np.array(angles) - ideal) ** 2) * weights)
        posture_score = params[0]
        return error + abs(posture_score - 50)

    def optimize_posture_score(self,angles, frame_num):
        if frame_num % self.OPTIMIZE_EVERY_N_FRAMES == 0:
            bounds = [(0, 100)]
            gwo = self.grey_wolf_optimizer(self.fitness_function, bounds, angles=angles)[0]
            goa = self.grasshopper_optimizer(self.fitness_function, bounds, angles=angles)[0]
            pso = self.particle_swarm_optimizer(self.fitness_function, bounds, angles=angles)[0]
            self.last_posture_score = (gwo + goa + pso) / 3
        return self.last_posture_score
    
    def fuse_posture_emotion(self,emotion_probs, posture_score, face_confidence):
        # Fixed weights: 40% for face emotion, 60% for posture
        w_face, w_posture = 0.3, 0.7
        # Initialize posture vector (8D: 7 emotions + 'No Emotion')
        posture_vec = [0.1] * 7 + [0.0]  # Default distribution
        if posture_score > 70:
            posture_vec[3], posture_vec[5], posture_vec[6] = 0.6, 0.2, 0.1  # Happy, Surprise, Neutral
        elif posture_score < 30:
            posture_vec[0], posture_vec[4], posture_vec[6] = 0.5, 0.3, 0.1  # Angry, Sad, Neutral

        # Fuse emotion probabilities and posture vector
        fused = w_face * np.array(emotion_probs) + w_posture * np.array(posture_vec)
        return fused, self.emotion_labels[np.argmax(fused)]

    def predict_emotion(self, video_path, output_path,optimizer="HYBRID"):
        optimizer_type = optimizer.upper()

        if optimizer_type != "HYBRID":
            print("❌ Only HYBRID is supported for fusion in this version.")
            sys.exit()

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        temp_output = output_path.replace(".mp4", "_temp.mp4")
        out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        lstm_sequence_buffer = deque(maxlen=self.LSTM_SEQUENCE_LENGTH)  # Buffer for LSTM sequences
        sequence_id = 0

        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                scale = 0.5
                resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = True

                # Pose detection
                pose_result = pose.process(img_rgb)
                back_angle = neck_angle = shoulder_angle = elbow_angle = 0
                posture_score = 50
                face_emotion = "No Emotion"
                fused_emotion = "No Emotion"
                pose_vec = [0] * 66  # Default pose vector

                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    try:
                        # Back angle: shoulder-hip-knee
                        shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                                    lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                        hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                        knee = [lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                        back_angle = self.calculate_angle(shoulder, hip, knee)

                        # Neck angle: ear-shoulder-hip
                        ear = [lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * width,
                            lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * height]
                        neck_angle = self.calculate_angle(ear, shoulder, hip)

                        # Shoulder angle: elbow-shoulder-hip
                        elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                        shoulder_angle = self.calculate_angle(elbow, shoulder, hip)

                        # Elbow angle: wrist-elbow-shoulder
                        wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                                lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                        elbow_angle = self.calculate_angle(wrist, elbow, shoulder)

                        angles = [back_angle, neck_angle, shoulder_angle, elbow_angle]
                        posture_score = self.optimize_posture_score(angles, frame_num)

                        # Extract 66D pose vector
                        pose_vec = self.extract_pose_vector(lm, width, height)
                        pose_entry = {"frame": frame_num}
                        for i, val in enumerate(pose_vec):
                            pose_entry[f"pose_{i}"] = round(val, 2)
                        self.pose_vector_log.append(pose_entry)
                    except Exception as e:
                        print(f"Angle calc error: {e}")

                    self.mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # YOLOv10 Face Detection
                face_img = None
                face_confidence = 0.0  # Default for no detection
                results = self.yolo_model(resized, conf=0.25, iou=0.7, imgsz=640)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    if len(boxes) > 0:
                        x1, y1, x2, y2 = boxes[0]  # First face
                        face_confidence = confidences[0]
                        x1, y1, x2, y2 = [int(coord / scale) for coord in [x1, y1, x2, y2]]
                        x, y = max(0, x1), max(0, y1)
                        w, h = min(x2 - x1, width - x), min(y2 - y1, height - y)
                        face_img = frame[y:y + h, x:x + w]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        break

                if face_img is not None and face_img.size != 0:
                    try:
                        face_input = self.preprocess_face(face_img)
                        preds = self.emotion_model.predict(face_input, verbose=0)[0]
                        face_emotion = self.emotion_labels[np.argmax(preds)]
                        preds = np.append(preds, 0.0)
                    except Exception as e:
                        print(f"Emotion prediction error: {e}")
                        face_emotion = "No Emotion"
                        preds = [0] * 8
                else:
                    preds = [0] * 8

                # Fuse and smooth emotions
                fused_probs, fused_emotion = self.fuse_posture_emotion(preds, posture_score, face_confidence)
                smoothed_probs = self.smooth_emotion_probs(fused_probs, self.emotion_probs_buffer)
                smoothed_emotion = self.emotion_labels[np.argmax(smoothed_probs)]

                # Prepare LSTM input sequence
                sequence_entry = {
                    "frame": frame_num,
                    "sequence_id": sequence_id,
                    **{f"emotion_prob_{i}": round(p, 4) for i, p in enumerate(fused_probs)},
                    **{f"pose_vec_{i}": round(v, 2) for i, v in enumerate(pose_vec)}
                }
                lstm_sequence_buffer.append(sequence_entry)
                if len(lstm_sequence_buffer) == self.LSTM_SEQUENCE_LENGTH:
                    for i, entry in enumerate(lstm_sequence_buffer):
                        self.lstm_input_log.append({
                            "sequence_id": sequence_id,
                            "frame": entry["frame"],
                            "timestep": i,
                            **{k: v for k, v in entry.items() if k not in ["frame", "sequence_id"]}
                        })
                    sequence_id += 1

                self.draw_text(frame, f"Back Angle: {int(back_angle)}", (30, 30))
                self.draw_text(frame, f"Neck Angle: {int(neck_angle)}", (30, 60))
                self.draw_text(frame, f"Shoulder Angle: {int(shoulder_angle)}", (30, 90))
                self.draw_text(frame, f"Elbow Angle: {int(elbow_angle)}", (30, 120))
                if back_angle < 60 or back_angle > 120:
                    self.draw_warning(frame, "Bad Back Posture!", (30, 150))
                if neck_angle < 40 or neck_angle > 90:
                    self.draw_warning(frame, "Bad Neck Posture!", (30, 180))

                self.draw_text(frame, f"Emotion (Face): {face_emotion}", (width - 280, 30), (255, 255, 255))
                self.draw_text(frame, f"Fused Emotion: {fused_emotion}", (width - 280, 60), (0, 255, 255))
                self.draw_text(frame, f"Smoothed Emotion: {smoothed_emotion}", (width - 280, 90), (255, 255, 0))
                self.draw_text(frame, f"Face Confidence: {face_confidence:.2f}", (width - 280, 120), (255, 255, 0))

                self.log_data.append({
                    "frame": frame_num,
                    "back_angle": back_angle,
                    "neck_angle": neck_angle,
                    "shoulder_angle": shoulder_angle,
                    "elbow_angle": elbow_angle,
                    "posture_score": round(posture_score, 2),
                    "face_emotion": face_emotion,
                    "fused_emotion": fused_emotion,
                    "smoothed_emotion": smoothed_emotion,
                    "face_confidence": face_confidence
                })

                out.write(frame)
                frame_num += 1
                

        cap.release()
        out.release()

        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)
            os.remove(temp_output)  # Remove temporary file after conversion
        except Exception as e:
            print("❌ Error during video processing:", e)

        df = pd.DataFrame(self.log_data)
        df.to_csv("emotion_posture_log.csv", index=False)
        print("✅ Saved emotion and posture log: emotion_posture_log.csv")

        pose_df = pd.DataFrame(self.pose_vector_log)
        pose_df.to_csv("pose_vector_log.csv", index=False)
        print("✅ Saved 66D pose vectors to: pose_vector_log.csv")

        lstm_df = pd.DataFrame(self.lstm_input_log)
        lstm_df.to_csv("lstm_input_log.csv", index=False)
        print("✅ Saved LSTM input sequences to: lstm_input_log.csv")

        if self.log_data:
            smoothed_emotions = [entry['smoothed_emotion'] for entry in self.log_data]
            final_emotion = Counter(smoothed_emotions).most_common(1)[0][0]
            print("✅ Final Smoothed Emotion (Majority Voting):", final_emotion)
        else:
            print("✅ Final Smoothed Emotion: No Emotion (No frames processed)")
        
        return final_emotion
        
    def post(self, request):
        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)
            
            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)
            
            video_uuid = request.session.get("video_uuid")
            if not video_uuid:
                return JsonResponse({"error": "No video UUID found in session"}, status=400)

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"posture_video_{video_uuid}.mp4"

            output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, output_filename)
            

            result = self.predict_emotion(video_absolute_path, output_path)


            return JsonResponse({
                'result': result,
                'status': 'success',
                'video_path': f"{settings.MEDIA_URL}output/{output_filename}"
            })
        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)

'''

################################ posture detection end ################################
class Eye(View):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn  = MTCNN(keep_all=True, device=device)

        self.previous_eye_center = None
        self.fixation_duration   = 0
        self.THRESHOLD           = 5
        self.emotion_counts      = Counter()
        self.shape_predictor = dlib.shape_predictor("/home/student/new_api/testpro-main/models/shape_predictor_68_face_landmarks.dat")

    def classify_eye_tracking(self, landmarks):
        left_eye  = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        lc = left_eye.mean(axis=0)
        rc = right_eye.mean(axis=0)
        pd = np.linalg.norm(lc - rc)

        cc = (lc + rc) * 0.5
        if self.previous_eye_center is not None:
            movement = np.linalg.norm(cc - self.previous_eye_center)
            self.fixation_duration = self.fixation_duration + 1 if movement < self.THRESHOLD else 0
        self.previous_eye_center = cc

        if   pd > 5 and self.fixation_duration < 10: 
            return "Focused"
        elif pd < 3 and self.fixation_duration < 10:
            return "Distracted"
        elif pd > 5 and self.fixation_duration > 20: 
            return "Intense"
        elif pd < 3 and self.fixation_duration > 20: 
            return "Nervous"
        else:                                  
            return "Natural"
    

    def process_frame(self, frame):
        # Detect faces & keypoints on GPU
        boxes, probs = self.mtcnn.detect(frame)
        emotion = "Unknown"

        if boxes is not None:
            for box in boxes.astype(int):
                x1,y1,x2,y2 = box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # Landmarks & classify
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(x1, y1, x2, y2)
                lm   = self.shape_predictor(gray, rect)
                emotion = self.classify_eye_tracking(lm)

                # Draw eye landmarks
                for p in range(36,48):
                    x,y = lm.part(p).x, lm.part(p).y
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        self.emotion_counts[emotion] += 1
        cv2.putText(frame, f"Eye Emotion: {emotion}", (20,40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        return frame

    def calculate_final_emotion(self):
        """Calculate the most frequent emotion and its percentage."""
        if not self.emotion_counts:
            return "Unknown", 0
            
        total_frames = sum(self.emotion_counts.values())
        most_common_emotion, count = self.emotion_counts.most_common(1)[0]
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        return most_common_emotion, percentage

    def process_video_parallel(self, input_path, output_path,
                            max_workers=8, batch_size=32):
        self.emotion_counts.clear()
        
        cap = cv2.VideoCapture(input_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Use H.264 codec
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_output, fourcc, fps, (w,h))

        executor = ThreadPoolExecutor(max_workers=max_workers)

        while True:
            batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append(frame)
            if not batch:
                break

            # Submit with index
            futures = {
                executor.submit(self.process_frame, frame): idx
                for idx, frame in enumerate(batch)
            }

            # Collect into a list and sort by original index
            results = []
            for fut in as_completed(futures):
                idx = futures[fut]
                annotated = fut.result()
                results.append((idx, annotated))

            for idx, annotated_frame in sorted(results, key=lambda x: x[0]):
                writer.write(annotated_frame)

        cap.release()
        writer.release()
        executor.shutdown(wait=True)

        # Convert to web-compatible format using FFmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            os.remove(temp_output)  # Clean up temporary file
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            # Fall back to temporary file if conversion fails
            os.rename(temp_output, output_path)

        final_emotion, percentage = self.calculate_final_emotion()
        return final_emotion, percentage

    def post(self, request):
        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            # Create output directory if it doesn't exist
            output_dir = os.path.join(settings.MEDIA_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"eye_tracking_{uuid.uuid4()}.mp4")
            final_emotion, percentage = self.process_video_parallel(video_absolute_path, output_path)

            # Return URL-friendly path
            relative_output_path = f"/output/{os.path.basename(output_path)}"
          
            return JsonResponse({
                'result': 'Eye tracking completed',
                'output_path': relative_output_path,  # Updated path format
                'final_emotion': final_emotion,
                'confidence': f"{percentage:.1f}",
                'status': 'success'
            })

        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)

# Facial
class HoneyBeeAlgorithm:
    """
    Honey Bee Algorithm for optimizing emotion detection parameters
    Features:
    - Adaptive confidence thresholds using bee foraging behavior
    - Dynamic facial region optimization through scout bee exploration
    - Multi-objective fitness evaluation balancing accuracy and stability
    - Swarm intelligence for real-time parameter tuning
    """

    def __init__(self, colony_size=20, max_iterations=50, limit=10):
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit  # Abandonment limit for scout bees
        self.employed_bees = colony_size // 2
        self.onlooker_bees = colony_size // 2
        
        # Initialize solution space for emotion detection parameters
        self.solutions = []
        self.fitness = []
        self.trial_counter = []
        
        # Parameter bounds for optimization
        self.bounds = {
            'face_confidence': (0.3, 0.9),
            'crop_scale_factor': (0.8, 1.4),
            'emotion_threshold': (0.4, 0.8),
            'temporal_smoothing': (0.1, 0.6),
            'detection_sensitivity': (0.5, 0.95)
        }
        
        self.initialize_population()
    
    def initialize_population(self):
        """Initialize bee colony with random solutions"""
        for _ in range(self.colony_size):
            solution = {
                'face_confidence': random.uniform(*self.bounds['face_confidence']),
                'crop_scale_factor': random.uniform(*self.bounds['crop_scale_factor']),
                'emotion_threshold': random.uniform(*self.bounds['emotion_threshold']),
                'temporal_smoothing': random.uniform(*self.bounds['temporal_smoothing']),
                'detection_sensitivity': random.uniform(*self.bounds['detection_sensitivity'])
            }
            self.solutions.append(solution)
            self.fitness.append(0.0)
            self.trial_counter.append(0)
    
    def calculate_fitness(self, solution, emotion_history, confidence_history):
        """Calculate fitness based on emotion detection consistency and confidence"""
        if len(emotion_history) < 3:
            return 0.5
        
        # Stability metric - prefer consistent detections
        recent_emotions = emotion_history[-10:]
        stability = 1.0 - (len(set(recent_emotions)) / len(recent_emotions))
        
        # Confidence metric - based on average detection confidence
        avg_confidence = np.mean(confidence_history[-10:]) if confidence_history else 0.5
        
        # Parameter balance metric
        param_balance = (solution['face_confidence'] + solution['emotion_threshold']) / 2
        
        # Multi-objective fitness combining stability, confidence, and parameter balance
        fitness = 0.4 * stability + 0.3 * avg_confidence + 0.3 * param_balance
        return min(max(fitness, 0.0), 1.0)
    
    def employed_bee_phase(self, emotion_history, confidence_history):
        """Employed bees search around current solutions"""
        for i in range(self.employed_bees):
            # Generate neighbor solution
            neighbor = self.solutions[i].copy()
            param = random.choice(list(self.bounds.keys()))
            
            # Random walk around current position
            step_size = random.uniform(-0.1, 0.1)
            neighbor[param] += step_size * (self.bounds[param][1] - self.bounds[param][0])
            neighbor[param] = max(min(neighbor[param], self.bounds[param][1]), self.bounds[param][0])
            
            # Evaluate fitness
            new_fitness = self.calculate_fitness(neighbor, emotion_history, confidence_history)
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.solutions[i] = neighbor
                self.fitness[i] = new_fitness
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1
    
    def onlooker_bee_phase(self, emotion_history, confidence_history):
        """Onlooker bees select solutions based on fitness probability"""
        fitness_sum = sum(self.fitness)
        if fitness_sum == 0:
            return
        
        for _ in range(self.onlooker_bees):
            # Roulette wheel selection
            prob = random.random()
            cumulative_prob = 0
            selected_idx = 0
            
            for i in range(self.employed_bees):
                cumulative_prob += self.fitness[i] / fitness_sum
                if prob <= cumulative_prob:
                    selected_idx = i
                    break
            
            # Search around selected solution
            neighbor = self.solutions[selected_idx].copy()
            param = random.choice(list(self.bounds.keys()))
            
            step_size = random.uniform(-0.05, 0.05)
            neighbor[param] += step_size * (self.bounds[param][1] - self.bounds[param][0])
            neighbor[param] = max(min(neighbor[param], self.bounds[param][1]), self.bounds[param][0])
            
            new_fitness = self.calculate_fitness(neighbor, emotion_history, confidence_history)
            
            if new_fitness > self.fitness[selected_idx]:
                self.solutions[selected_idx] = neighbor
                self.fitness[selected_idx] = new_fitness
                self.trial_counter[selected_idx] = 0
    
    def scout_bee_phase(self):
        """Scout bees abandon poor solutions and explore new areas"""
        for i in range(self.colony_size):
            if self.trial_counter[i] > self.limit:
                # Generate new random solution
                solution = {
                    'face_confidence': random.uniform(*self.bounds['face_confidence']),
                    'crop_scale_factor': random.uniform(*self.bounds['crop_scale_factor']),
                    'emotion_threshold': random.uniform(*self.bounds['emotion_threshold']),
                    'temporal_smoothing': random.uniform(*self.bounds['temporal_smoothing']),
                    'detection_sensitivity': random.uniform(*self.bounds['detection_sensitivity'])
                }
                self.solutions[i] = solution
                self.fitness[i] = 0.0
                self.trial_counter[i] = 0
    
    def get_best_solution(self):
        """Return the best solution found by the colony"""
        best_idx = max(range(len(self.fitness)), key=lambda i: self.fitness[i])
        return self.solutions[best_idx], self.fitness[best_idx]
    
    def optimize(self, emotion_history, confidence_history):
        """Run one iteration of the HBA optimization"""
        self.employed_bee_phase(emotion_history, confidence_history)
        self.onlooker_bee_phase(emotion_history, confidence_history)
        self.scout_bee_phase()

# ------------------- Facial Emotion Detection class -------------------
class Face(View):
    def __init__(self):
        super().__init__()

        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize models
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mtcnn = MTCNN(device=device)
        self.model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to(device)
        self.extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")

    def calculate_angle(self, a, b, c):
        """Calculates the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    def detect_emotions_hba_optimized(self, frame, hba_params):
        """
        HBA-Enhanced emotion detection with adaptive parameters
        Features: Swarm intelligence optimized face detection and emotion classification
        """
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces, probs = self.mtcnn.detect(img)

        if faces is None or len(faces) == 0:
            return "Neutral", 0.3  # Default to neutral if no face is detected

        # Apply HBA-optimized face confidence threshold
        face_confidence = probs[0] if probs is not None else 0.5
        if face_confidence < hba_params['face_confidence']:
            return "Neutral", face_confidence

        # HBA-optimized face cropping with dynamic scaling
        face_box = faces[0]
        scale_factor = hba_params['crop_scale_factor']
        
        # Calculate expanded face region using HBA parameters
        center_x = (face_box[0] + face_box[2]) / 2
        center_y = (face_box[1] + face_box[3]) / 2
        width = (face_box[2] - face_box[0]) * scale_factor
        height = (face_box[3] - face_box[1]) * scale_factor
        
        # Ensure bounds are within image
        x1 = max(0, int(center_x - width/2))
        y1 = max(0, int(center_y - height/2))
        x2 = min(img.width, int(center_x + width/2))
        y2 = min(img.height, int(center_y + height/2))
        
        face = img.crop((x1, y1, x2, y2))
        
        # Process emotion detection
        inputs = self.extractor(images=face, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        emotion_confidence = torch.max(probs).item()
        
        # Apply HBA-optimized emotion threshold
        if emotion_confidence < hba_params['emotion_threshold']:
            return "Neutral", emotion_confidence
        
        emotion = self.model.config.id2label[torch.argmax(probs).item()].capitalize()
        
        return emotion, emotion_confidence

    def classify_posture(self, back_angle, neck_angle):
        """Classifies posture based on back and neck angles."""
        if back_angle > 170 and neck_angle > 150:
            return "Confident"
        elif back_angle < 160 and neck_angle < 140:
            return "Nervous"
        elif back_angle < 150:
            return "Defensive"
        elif neck_angle < 130:
            return "Serious"
        else:
            return "Attentive"
        
    def apply_hba_temporal_smoothing(self, emotion_history, current_emotion, smoothing_factor):
        """
        HBA-optimized temporal smoothing for emotion stability
        Feature: Bee colony consensus-based emotion filtering
        """
        if len(emotion_history) < 3:
            return current_emotion
        
        # Analyze recent emotion patterns
        recent_emotions = emotion_history[-7:]
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Apply HBA-based consensus mechanism
        if len(emotion_counts) == 1:
            # Single emotion detected - high confidence
            return current_emotion
        
        # Multiple emotions - apply bee colony consensus
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_count = emotion_counts[dominant_emotion]
        
        # HBA consensus threshold based on smoothing factor
        consensus_threshold = len(recent_emotions) * smoothing_factor
        
        if dominant_count >= consensus_threshold:
            return dominant_emotion
        else:
            return current_emotion

    def process_video_hba_enhanced(self, input_path):
        """
        HBA-Enhanced video processing with your original structure
        Key Features:
        - Maintains your existing emotion detection model
        - Adds swarm intelligence parameter optimization
        - Real-time adaptive thresholding using bee foraging patterns
        - Colony consensus temporal smoothing
        - Scout bee exploration for robust detection
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize Honey Bee Algorithm
        hba = HoneyBeeAlgorithm(colony_size=20, max_iterations=50)
        
        emotion_counts = {}
        emotion_history = []
        confidence_history = []
        total_frames = 0
        
        print("🐝 HBA-Enhanced Emotion Detection Started")
        print("📊 Swarm intelligence optimization active...")
        print("🎯 Adaptive parameter tuning with your original ML model...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_frame)

            # Posture analysis (keeping your original logic)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
                ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y]

                back_angle = self.calculate_angle(shoulder, hip, knee)
                neck_angle = self.calculate_angle(ear, shoulder, hip)
                posture_label = self.classify_posture(back_angle, neck_angle)
            else:
                posture_label = "Unknown"

            # Get current best HBA parameters
            best_params, fitness = hba.get_best_solution()
            
            # HBA-enhanced emotion detection
            emotion, confidence = self.detect_emotions_hba_optimized(frame, best_params)

            # Apply HBA temporal smoothing
            smoothed_emotion = self.apply_hba_temporal_smoothing(
                emotion_history, emotion, best_params['temporal_smoothing']
            )
            
            emotion_history.append(smoothed_emotion)
            confidence_history.append(confidence)
            emotion_counts[smoothed_emotion] = emotion_counts.get(smoothed_emotion, 0) + 1
            
            # Update HBA optimization every 15 frames
            if total_frames % 15 == 0:
                hba.optimize(emotion_history, confidence_history)
            
            # Display progress every second
            if total_frames % fps == 0:
                major_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Unknown"
                major_emotion_percent = (emotion_counts[major_emotion] / total_frames) * 100
                print(f"🐝 Processing: {total_frames}/{total_video_frames} frames | "
                    f"Major Emotion: {major_emotion} ({major_emotion_percent:.1f}%) | "
                    f"HBA Fitness: {fitness:.3f}")

        cap.release()
        cv2.destroyAllWindows()
        
        # Final results with HBA optimization summary
        if emotion_counts:
            major_emotion = max(emotion_counts, key=emotion_counts.get)
            major_emotion_percent = (emotion_counts[major_emotion] / total_frames) * 100
            major_emotion_frames = emotion_counts[major_emotion]
            
            print("\n" + "="*65)
            print("🐝 HBA-ENHANCED EMOTION ANALYSIS COMPLETE")
            print("="*65)
            print(f"📈 MAJOR EMOTION: {major_emotion}")
            print(f"📊 PERCENTAGE: {major_emotion_percent:.2f}%")
            print(f"🎬 FRAME COUNT: {major_emotion_frames}/{total_frames}")
            print(f"🧠 FINAL HBA FITNESS: {fitness:.3f}")
            print("="*65)
            
            print("\n🐝 HBA OPTIMIZATION FEATURES INTEGRATED:")
            print("✅ Swarm intelligence parameter optimization")
            print("✅ Adaptive face detection confidence thresholding")
            print("✅ Dynamic facial region scaling optimization") 
            print("✅ Bee colony consensus temporal smoothing")
            print("✅ Multi-objective fitness evaluation")
            print("✅ Scout bee exploration for detection robustness")
            print("✅ Real-time parameter tuning during video processing")
            
            return {
                'major_emotion': major_emotion,
                'percentage': major_emotion_percent,
                'frame_count': major_emotion_frames,
                'total_frames': total_frames,
                'hba_fitness': fitness,
                'all_emotions': emotion_counts
            }
        else:
            print("❌ No emotions detected in the video")
            return None

    def post(self, request):
        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            print("\n--------------------HBA Enhanced Emotion Detection---------------------\n")
            result = self.process_video_hba_enhanced(video_absolute_path)
            print("\n--------------------HBA Enhanced Emotion Detection---------------------\n")

            if result:
                return JsonResponse({
                    'result': 'HBA-Enhanced emotion detection completed',
                    'major_emotion': result['major_emotion'],
                    'percentage': f"{result['percentage']:.1f}",
                    'frame_count': result['frame_count'],
                    'total_frames': result['total_frames'],
                    'hba_fitness': f"{result['hba_fitness']:.3f}",
                    'all_emotions': result['all_emotions'],
                    'status': 'success'
                })
            else:
                return JsonResponse({"error": "No emotions detected in the video"}, status=404)

        except FileNotFoundError as e:
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)


########################## Audio Tone starts from here ##############################

'''# --- Global Configuration Fallbacks & Constants ---
class FallbackPreprocConfigRef:
    OTHER_FEATURE_KEYS_ORDERED = ['mfcc','f0','rms','chroma','jitter','shimmer','hnr','f1','f2','f3']
    N_MFCC=20; N_CHROMA=12; MAX_LEN_OTHER_FEATURES_SEQ=32
    OTHER_FEATURE_DIMS = {'mfcc':N_MFCC,'f0':1,'rms':1,'chroma':N_CHROMA,'jitter':1,'shimmer':1,'hnr':1,'f1':1,'f2':1,'f3':1}
    SAMPLE_RATE=16000; N_MELS=80; MAX_LEN_SPECTROGRAM=128
    EMOTIONS = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful"}

# --- Reproducibility ---
def set_seed(seed_value):
    torch.manual_seed(seed_value); np.random.seed(seed_value); random.seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

# --- Analysis Script Configuration ---
class AnalysisConfig:
    GG_RUN_OUTPUT_DIR = "/home/student/new_api/output_gg_v4.9_6class_full"
    EMOTIONS = {}; NUM_CLASSES = 0
    SAMPLE_RATE = FallbackPreprocConfigRef.SAMPLE_RATE
    N_MELS = FallbackPreprocConfigRef.N_MELS
    MAX_LEN_SPECTROGRAM = FallbackPreprocConfigRef.MAX_LEN_SPECTROGRAM
    MODEL_ARCH_CONSTRUCTOR = []
    SELECTED_FEATURE_KEYS_OTHER = []
    OTHER_FEATURES_TOTAL_DIM_CURRENT = 0
    _FEATURE_DIMS_OTHER_DEFAULT = dict(FallbackPreprocConfigRef.OTHER_FEATURE_DIMS)
    MAX_LEN_OTHER_FEATURES_SEQ = FallbackPreprocConfigRef.MAX_LEN_OTHER_FEATURES_SEQ
    SEGMENT_DURATION_S = 4.0
    SEGMENT_OVERLAP_S = 1.0
    SEED = 42

    def __init__(self):
        self.EFFECTIVE_GG_CONFIG_LOAD_PATH = os.path.join(self.GG_RUN_OUTPUT_DIR, "config_run_settings.yaml")
        self.MODEL_SAVE_PATH = ""
        self.NORM_STATS_SAVE_PATH = ""

    def load_parameters_from_gg_config(self):
        if not os.path.exists(self.EFFECTIVE_GG_CONFIG_LOAD_PATH):
            print(f"CRITICAL ERROR: gg.py config file NOT FOUND at {self.EFFECTIVE_GG_CONFIG_LOAD_PATH}. Exiting.")
            sys.exit(1)
        try:
            with open(self.EFFECTIVE_GG_CONFIG_LOAD_PATH, 'r') as f: loaded_gg_cfg = yaml.safe_load(f)
            missing_keys = []
            if 'MODEL_ARCH_CONSTRUCTOR_EFFECTIVE' in loaded_gg_cfg: self.MODEL_ARCH_CONSTRUCTOR = loaded_gg_cfg['MODEL_ARCH_CONSTRUCTOR_EFFECTIVE']
            else: missing_keys.append("MODEL_ARCH_CONSTRUCTOR_EFFECTIVE")
            if 'EFFECTIVE_SELECTED_FEATURE_KEYS' in loaded_gg_cfg: self.SELECTED_FEATURE_KEYS_OTHER = list(loaded_gg_cfg['EFFECTIVE_SELECTED_FEATURE_KEYS'])
            else: missing_keys.append("EFFECTIVE_SELECTED_FEATURE_KEYS")
            if 'EFFECTIVE_OTHER_FEATURES_DIM' in loaded_gg_cfg: self.OTHER_FEATURES_TOTAL_DIM_CURRENT = loaded_gg_cfg['EFFECTIVE_OTHER_FEATURES_DIM']
            elif self.SELECTED_FEATURE_KEYS_OTHER: self.OTHER_FEATURES_TOTAL_DIM_CURRENT = sum(self._FEATURE_DIMS_OTHER_DEFAULT.get(k,1) for k in self.SELECTED_FEATURE_KEYS_OTHER)
            else: missing_keys.append("EFFECTIVE_OTHER_FEATURES_DIM")
            if 'EMOTIONS' in loaded_gg_cfg: self.EMOTIONS = loaded_gg_cfg['EMOTIONS']; self.NUM_CLASSES = len(self.EMOTIONS)
            else: missing_keys.append("EMOTIONS")
            if missing_keys: print(f"CRITICAL ERROR: Keys missing in YAML: {', '.join(missing_keys)}. Exiting."); sys.exit(1)
            self.SAMPLE_RATE = loaded_gg_cfg.get('SAMPLE_RATE', self.SAMPLE_RATE)
            self.N_MELS = loaded_gg_cfg.get('N_MELS', self.N_MELS)
            self.MAX_LEN_SPECTROGRAM = loaded_gg_cfg.get('MAX_LEN_SPECTROGRAM', self.MAX_LEN_SPECTROGRAM)
            self.MAX_LEN_OTHER_FEATURES_SEQ = loaded_gg_cfg.get('MAX_LEN_OTHER_FEATURES_SEQ', self.MAX_LEN_OTHER_FEATURES_SEQ)
            self.SEED = loaded_gg_cfg.get('SEED', self.SEED)
            model_subdir = loaded_gg_cfg.get('MODEL_SUBDIR', 'saved_models')
            model_filename = loaded_gg_cfg.get('FINAL_MODEL_FILE_NAME', 'audio_emotion_final_model.pth')
            norm_stats_filename = loaded_gg_cfg.get('NORM_STATS_FILE_NAME', 'feature_norm_stats.pt')
            self.MODEL_SAVE_PATH = os.path.join(self.GG_RUN_OUTPUT_DIR, model_subdir, model_filename)
            self.NORM_STATS_SAVE_PATH = os.path.join(self.GG_RUN_OUTPUT_DIR, model_subdir, norm_stats_filename)
            if 'FEATURE_DIMS_OTHER_DEFAULT' in loaded_gg_cfg: self._FEATURE_DIMS_OTHER_DEFAULT = loaded_gg_cfg['FEATURE_DIMS_OTHER_DEFAULT']
        except Exception as e: print(f"CRITICAL ERROR: Problem loading training config: {e}"); sys.exit(1)

cfg_analyzer = AnalysisConfig()

FILTER_CHOICES_CRNN_NAS, LSTM_HIDDEN_CHOICES_NAS, DENSE_CHOICES_CRNN_NAS = [32,64,96,128], [64,128,256,384], [128,256,512]
NAS_BOUNDS_CRNN_FALLBACK = [(1,3),(0,1),(0,3),(0,1),(0,3),(0,1),(0,3),(0,0),(0.1,0.5),(0,3),(1,2),(0.1,0.4),(0,2),(5e-5,1e-3),(0,1)]
class DynamicAudioCRNN(nn.Module):
    def __init__(self, spec_input_shape_hw, current_other_features_dim, arch_params_constructor):
        super(DynamicAudioCRNN, self).__init__()
        self.spec_input_shape_hw = spec_input_shape_hw; self.current_other_features_dim = current_other_features_dim
        self.cnn_layers = nn.Sequential(); self.last_conv_layer_name_for_gradcam = None
        p_idx, in_cnn_channels = 0, 1
        current_nas_bounds = NAS_BOUNDS_CRNN_FALLBACK
        try:
            num_conv=int(arch_params_constructor[p_idx]); p_idx+=1
            act_c1=int(arch_params_constructor[p_idx]); p_idx+=1
            f1_idx=int(arch_params_constructor[p_idx]);p_idx+=1;f1=FILTER_CHOICES_CRNN_NAS[f1_idx]
            k1_idx=int(arch_params_constructor[p_idx]);p_idx+=1;k1=3 if k1_idx==0 else 5
            conv1_name="conv1";self.cnn_layers.add_module(conv1_name,nn.Conv2d(in_cnn_channels,f1,k1,padding=k1//2))
            self.cnn_layers.add_module("bn1",nn.BatchNorm2d(f1));self.cnn_layers.add_module("act1",nn.ReLU()if act_c1==0 else nn.LeakyReLU(0.1))
            self.cnn_layers.add_module("pool1",nn.MaxPool2d(2,2));cnn_out_ch=f1;self.last_conv_layer_name_for_gradcam=conv1_name
            if num_conv >= 2:
                f2_idx=int(arch_params_constructor[p_idx]);p_idx+=1;f2=FILTER_CHOICES_CRNN_NAS[f2_idx]
                k2_idx=int(arch_params_constructor[p_idx]);p_idx+=1;k2=3 if k2_idx==0 else 5
                conv2_name="conv2";self.cnn_layers.add_module(conv2_name,nn.Conv2d(cnn_out_ch,f2,k2,padding=k2//2))
                self.cnn_layers.add_module("bn2",nn.BatchNorm2d(f2));self.cnn_layers.add_module("act2",nn.ReLU())
                self.cnn_layers.add_module("pool2",nn.MaxPool2d(2,2));cnn_out_ch=f2;self.last_conv_layer_name_for_gradcam=conv2_name
            if num_conv == 3:
                f3_idx=int(arch_params_constructor[p_idx]);p_idx+=1;f3=FILTER_CHOICES_CRNN_NAS[f3_idx]
                k3_idx_is_choice = len(current_nas_bounds) > 7 and current_nas_bounds[7][1] > current_nas_bounds[7][0]
                if len(arch_params_constructor) > p_idx :
                    k3_idx_param=int(arch_params_constructor[p_idx]); p_idx+=1; k3=3 if k3_idx_param==0 else 5
                elif k3_idx_is_choice: p_idx+=1; k3=3
                else: k3=3
                conv3_name="conv3";self.cnn_layers.add_module(conv3_name,nn.Conv2d(cnn_out_ch,f3,k3,padding=k3//2))
                self.cnn_layers.add_module("bn3",nn.BatchNorm2d(f3));self.cnn_layers.add_module("act3",nn.ReLU())
                self.cnn_layers.add_module("pool3",nn.MaxPool2d(2,2));cnn_out_ch=f3;self.last_conv_layer_name_for_gradcam=conv3_name
            with torch.no_grad():dummy_spec=torch.zeros(1,1,*self.spec_input_shape_hw);cnn_out_dum=self.cnn_layers(dummy_spec)
            self.cnn_feat_dim_lstm=cnn_out_dum.shape[1]*cnn_out_dum.shape[2];self.cnn_seq_len_lstm=cnn_out_dum.shape[3]
            conv_drop=float(arch_params_constructor[p_idx]);p_idx+=1;self.cnn_dropout=nn.Dropout(conv_drop)
            lstm_h_idx=int(arch_params_constructor[p_idx]);p_idx+=1;lstm_h=LSTM_HIDDEN_CHOICES_NAS[lstm_h_idx]
            num_lstm_l=int(arch_params_constructor[p_idx]);p_idx+=1
            lstm_dropout_param=arch_params_constructor[p_idx];p_idx+=1
            try:
                lstm_dropout_value=float(lstm_dropout_param)
                if not(0.0<=lstm_dropout_value<=0.5):lstm_dropout_value=np.clip(lstm_dropout_value,0.0,0.5)
            except Exception: lstm_dropout_value=0.1
            actual_lstm_dropout=lstm_dropout_value if num_lstm_l>1 else 0.0
            self.lstm_input_dim=self.cnn_feat_dim_lstm
            if self.current_other_features_dim>0:self.lstm_input_dim+=self.current_other_features_dim
            self.lstm=nn.LSTM(self.lstm_input_dim,lstm_h,num_lstm_l,batch_first=True,dropout=actual_lstm_dropout,bidirectional=True)
            d1_idx=int(arch_params_constructor[p_idx]);p_idx+=1;d1=DENSE_CHOICES_CRNN_NAS[d1_idx]
            if cfg_analyzer.NUM_CLASSES == 0: raise ValueError("NUM_CLASSES not set for model output layer.")
            self.fc_layers=nn.Sequential(nn.Linear(lstm_h*2,d1),nn.ReLU(),nn.Dropout(0.5),nn.Linear(d1,cfg_analyzer.NUM_CLASSES))
        except IndexError: print(f"ERROR CRNN Arch IdxErr: p_idx={p_idx},arch_len={len(arch_params_constructor)},arch:{arch_params_constructor}."); raise
        except Exception as e: print(f"ERROR CRNN Arch Err: {e},arch:{arch_params_constructor}"); raise
    def forward(self,x_tuple):
        x_s,x_o=x_tuple
        if x_s.dim()==3:x_s=x_s.unsqueeze(1)
        cnn_o=self.cnn_layers(x_s);cnn_o=self.cnn_dropout(cnn_o)
        cnn_lstm_in=cnn_o.permute(0,3,1,2).contiguous().view(cnn_o.size(0),self.cnn_seq_len_lstm,-1)
        combined_in=cnn_lstm_in
        if self.current_other_features_dim>0 and x_o is not None and x_o.numel()>0 and x_o.shape[1]>0:
            other_p=x_o.permute(0,2,1)
            if other_p.shape[1]!=self.cnn_seq_len_lstm:
                other_p=nn.functional.interpolate(other_p.permute(0,2,1),size=self.cnn_seq_len_lstm,mode='linear',align_corners=False).permute(0,2,1)
            combined_in=torch.cat((cnn_lstm_in,other_p),dim=2)
        lstm_out,_=self.lstm(combined_in);last_step=lstm_out[:,-1,:];final_out=self.fc_layers(last_step)
        return final_out

def _pad_or_truncate_feature_pred(feat_array_orig, target_len_frames, num_coeffs):
    feat_array = np.copy(feat_array_orig);
    if not isinstance(feat_array, np.ndarray) or feat_array.size == 0: return np.zeros((num_coeffs, target_len_frames), dtype=np.float32)
    if feat_array.ndim == 1: feat_array = feat_array.reshape(1, -1)
    if feat_array.shape[0] != num_coeffs:
        if feat_array.shape[0] < num_coeffs: feat_array = np.pad(feat_array, ((0,num_coeffs-feat_array.shape[0]),(0,0)),mode='constant', constant_values=0)
        else: feat_array = feat_array[:num_coeffs,:]
    current_time_len = feat_array.shape[1]
    if current_time_len == 0: return np.zeros((num_coeffs, target_len_frames), dtype=np.float32)
    if current_time_len > target_len_frames: return feat_array[:,:target_len_frames].astype(np.float32)
    elif current_time_len < target_len_frames: return np.pad(feat_array, ((0,0),(0,target_len_frames-current_time_len)),mode='constant', constant_values=0).astype(np.float32)
    return feat_array.astype(np.float32)

def extract_librosa_features_for_pred(waveform_np, sr, target_len_frames):
    hop_len=512; N_MFCC=cfg_analyzer._FEATURE_DIMS_OTHER_DEFAULT.get('mfcc',20); N_CHROMA=cfg_analyzer._FEATURE_DIMS_OTHER_DEFAULT.get('chroma',12)
    mfccs=librosa.feature.mfcc(y=waveform_np,sr=sr,n_mfcc=N_MFCC,hop_length=hop_len)
    f0,_,_=librosa.pyin(waveform_np,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),hop_length=hop_len, fill_na=0.0)
    rms=librosa.feature.rms(y=waveform_np,hop_length=hop_len)[0].reshape(1,-1)
    chroma=librosa.feature.chroma_stft(y=waveform_np,sr=sr,n_chroma=N_CHROMA,hop_length=hop_len)
    raw_feats={'mfcc':mfccs,'f0':f0,'rms':rms,'chroma':chroma}; processed_feats={}
    for key,feat_arr in raw_feats.items():
        num_c=cfg_analyzer._FEATURE_DIMS_OTHER_DEFAULT.get(key,1)
        processed_feats[key]=_pad_or_truncate_feature_pred(feat_arr,target_len_frames,num_c)
    return processed_feats

def extract_detailed_acoustic_features_for_pred(sound_obj, target_len_frames):
    processed_praat_feats={}; fallback_dims=cfg_analyzer._FEATURE_DIMS_OTHER_DEFAULT
    def get_zeros_praat(): return {k:np.zeros((fallback_dims.get(k,1),target_len_frames),dtype=np.float32) for k in ['jitter','shimmer','hnr','f1','f2','f3']}
    try:
        if getattr(sound_obj,'duration',0) < 0.02: return get_zeros_praat()
        pitch = sound_obj.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0, very_accurate=False)
        if pitch is None or getattr(pitch,'n_frames',0)==0: return get_zeros_praat()
        pp=parselmouth.praat.call(pitch,"To PointProcess (cc)"); jitter=parselmouth.praat.call(pp,"Get jitter (local)",0,0,0.0001,0.02,1.3)*100
        shimmer=parselmouth.praat.call(pp,"Get shimmer (local)",0,0,0.0001,0.02,1.6)*100
        h_obj=sound_obj.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0); hnr_vals=[h_obj.get_value(t) for t in h_obj.ts() if h_obj.get_value(t) is not None]
        hnr=np.nan_to_num(np.array(hnr_vals),nan=0.0) if hnr_vals else np.array([0.0])
        f_obj=sound_obj.to_formant_burg(time_step=0.01,max_number_of_formants=5,maximum_formant=5500,window_length=0.025);f1,f2,f3=[],[],[]
        for t in f_obj.ts():f1.append(f_obj.get_value_at_time(1,t));f2.append(f_obj.get_value_at_time(2,t));f3.append(f_obj.get_value_at_time(3,t))
        f1np=np.nan_to_num(np.array(f1),nan=0.0) if f1 else np.array([0.0]);f2np=np.nan_to_num(np.array(f2),nan=0.0) if f2 else np.array([0.0]);f3np=np.nan_to_num(np.array(f3),nan=0.0) if f3 else np.array([0.0])
        ref_len=len(hnr) if len(hnr)>1 else 10
        raw_praat={'jitter':np.full((1,ref_len),jitter if not np.isnan(jitter) else 0.0),'shimmer':np.full((1,ref_len),shimmer if not np.isnan(shimmer) else 0.0),'hnr':hnr,'f1':f1np,'f2':f2np,'f3':f3np}
        for key,arr in raw_praat.items(): processed_praat_feats[key]=_pad_or_truncate_feature_pred(arr,target_len_frames,fallback_dims.get(key,1))
        return processed_praat_feats
    except Exception: return get_zeros_praat()

def extract_features_for_segment(waveform_segment_np, sr, norm_stats, selected_feature_keys_list):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); waveform_segment_np=waveform_segment_np.astype(np.float32)
    wf_tensor=torch.tensor(waveform_segment_np).unsqueeze(0).to(device)
    if wf_tensor.ndim>2 and wf_tensor.shape[1]>1: wf_tensor=torch.mean(wf_tensor,dim=1,keepdim=True)
    elif wf_tensor.ndim==1: wf_tensor=wf_tensor.unsqueeze(0)

    spec_tr=nn.Sequential(torchaudio.transforms.MelSpectrogram(sr,n_mels=cfg_analyzer.N_MELS,n_fft=400,hop_length=160),torchaudio.transforms.AmplitudeToDB()).to(device)
    mel_spec=spec_tr(wf_tensor);
    mel_interp=nn.functional.interpolate(mel_spec.unsqueeze(1) if mel_spec.dim()==3 else mel_spec,size=(cfg_analyzer.N_MELS,cfg_analyzer.MAX_LEN_SPECTROGRAM),mode='bilinear',align_corners=False)
    if mel_interp.dim()==4 and mel_interp.shape[1]==1: mel_interp=mel_interp.squeeze(1)

    if norm_stats and 'spectrogram' in norm_stats:
        spec_s=norm_stats['spectrogram'];mean_s,std_s=spec_s['mean'].to(device),spec_s['std'].to(device)
        if mean_s.ndim==0:mean_s=mean_s.view(1,1,1);std_s=std_s.view(1,1,1)
        mel_interp=(mel_interp-mean_s)/(std_s+1e-7)

    list_of_normalized_acoustic_features = []
    concatenated_acoustic_features = torch.empty((mel_interp.shape[0],0,cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ),dtype=torch.float32).to(device)

    if selected_feature_keys_list and len(selected_feature_keys_list) > 0 :
        wf_np_1d=waveform_segment_np.squeeze();
        if wf_np_1d.ndim>1: wf_np_1d=np.mean(wf_np_1d,axis=0)

        lib_feats_p=extract_librosa_features_for_pred(wf_np_1d,sr,cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ); praat_feats_p={}
        try:
            if len(wf_np_1d)>sr*0.02: sound_p=parselmouth.Sound(wf_np_1d,sr);praat_feats_p=extract_detailed_acoustic_features_for_pred(sound_p,cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ)
        except Exception: pass

        for key_idx, key in enumerate(selected_feature_keys_list):
            num_c=cfg_analyzer._FEATURE_DIMS_OTHER_DEFAULT.get(key,1)
            raw_f_np = lib_feats_p.get(key)
            if raw_f_np is None:
                raw_f_np = praat_feats_p.get(key)
            if raw_f_np is None:
                raw_f_np = np.zeros((num_c,cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ),dtype=np.float32)

            if not isinstance(raw_f_np, np.ndarray):
                raw_f_np = np.zeros((num_c, cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ), dtype=np.float32)

            f_tensor=torch.tensor(raw_f_np,dtype=torch.float32).to(device)
            if f_tensor.ndim == 1 and num_c == 1 :
                f_tensor = f_tensor.unsqueeze(0)

            if norm_stats and key in norm_stats:
                f_s=norm_stats[key];mean_o,std_o=f_s['mean'].to(device),f_s['std'].to(device)
                if mean_o.ndim==1 and f_tensor.ndim==2 :mean_o=mean_o.unsqueeze(-1)
                if std_o.ndim==1 and f_tensor.ndim==2 :std_o=std_o.unsqueeze(-1)
                f_tensor=(f_tensor-mean_o)/(std_o+1e-7)
            list_of_normalized_acoustic_features.append(f_tensor)

        if list_of_normalized_acoustic_features:
            valid_tensors_to_cat = []
            for i, tnsr in enumerate(list_of_normalized_acoustic_features):
                # current_key = selected_feature_keys_list[i] # Not used, can be removed
                if tnsr is not None and isinstance(tnsr, torch.Tensor) and tnsr.numel() > 0:
                    if tnsr.ndim >= 2:
                        valid_tensors_to_cat.append(tnsr)
                    elif tnsr.ndim == 1:
                        valid_tensors_to_cat.append(tnsr.unsqueeze(0))
            if valid_tensors_to_cat:
                try:
                    stacked_other=torch.cat(valid_tensors_to_cat,0)
                    concatenated_acoustic_features=stacked_other.unsqueeze(0)
                except Exception as e_cat:
                    print(f"ERROR during torch.cat of acoustic features: {e_cat}")
    return (mel_interp, concatenated_acoustic_features)

def generate_simple_explanation(selected_features):
    if not selected_features: return "Model primarily analyzed overall sound patterns in the spectrogram."
    feature_categories = {
        'spectral shape': ['mfcc'], 'pitch': ['f0'], 'intensity': ['rms'], 'timbre/harmony': ['chroma'],
        'prosodic stability': ['jitter', 'shimmer'], 'voice quality': ['hnr'], 'vowel clarity': ['f1', 'f2', 'f3']
    }
    used_categories = set()
    for feat in selected_features:
        for cat, members in feature_categories.items():
            if feat in members: used_categories.add(cat.title()); break
    if not used_categories and 'mfcc' not in selected_features : explanation = "Model analyzed general acoustic patterns from the spectrogram."
    elif not used_categories and 'mfcc' in selected_features: explanation = "Model primarily analyzed spectral shape (MFCCs) and spectrogram patterns."
    else: explanation = f"Model considered spectrogram patterns and acoustic cues from: {', '.join(sorted(list(used_categories)))}."
    return explanation

# --- Core Video Analysis Function (Cleaner Output) ---
def analyze_video(current_video_path, model, norm_stats, device, selected_feature_keys_for_explanation):
    video_basename = os.path.basename(current_video_path)
    if not os.path.exists(current_video_path): return [], "Video file not found.", ""

    list_of_segment_predictions = []
    video_clip = None # Initialize to ensure it's defined for finally block
    audio_clip_mvp = None # Moviepy audio clip
    temp_audio_path = "temp_audio_for_analysis.wav" # In /content/

    try:
        video_clip = VideoFileClip(current_video_path)
        audio_clip_mvp = video_clip.audio

        if audio_clip_mvp is None:
            return [], "No audio in video.", ""

        if audio_clip_mvp.duration is None or audio_clip_mvp.duration < 0.02: # Check for very short or no duration
            print(f"Warning: Audio clip for {video_basename} has negligible or no duration ({audio_clip_mvp.duration}s). Skipping.")
            return [], "Audio duration too short or invalid.", ""

        # ***** START: WAV Workaround *****
        print(f"INFO: Writing temporary WAV for {video_basename} to {temp_audio_path}")
        audio_clip_mvp.write_audiofile(temp_audio_path,
                                       fps=cfg_analyzer.SAMPLE_RATE,
                                       codec='pcm_s16le', # Standard 16-bit WAV
                                       logger=None) # Suppress moviepy progress bar for this

        # Load with librosa
        print(f"INFO: Loading temporary WAV with librosa for {video_basename}")
        audio_data_mono, sr_read = librosa.load(temp_audio_path, sr=cfg_analyzer.SAMPLE_RATE, mono=True)

        if sr_read != cfg_analyzer.SAMPLE_RATE:
            print(f"Warning: Librosa loaded audio with sr {sr_read}, expected {cfg_analyzer.SAMPLE_RATE}. Resampling if necessary or check config.")
            # Librosa should resample if sr is specified and different, but good to be aware.

        # audio_data_mono is already float32 and mono from librosa.load
        # No need for: audio_data_mono = audio_data_mono.astype(np.float32)
        # ***** END: WAV Workaround *****

        seg_len_smp=int(cfg_analyzer.SEGMENT_DURATION_S*cfg_analyzer.SAMPLE_RATE)
        overlap_smp=int(cfg_analyzer.SEGMENT_OVERLAP_S*cfg_analyzer.SAMPLE_RATE)
        step_smp=seg_len_smp-overlap_smp

        if step_smp <= 0:
            return [], "Invalid segment step (duration too short or overlap too large).", ""

        num_segs=0
        if len(audio_data_mono) < seg_len_smp:
            if len(audio_data_mono) < cfg_analyzer.SAMPLE_RATE * 0.1: # e.g. 0.1 seconds
                return [], "Audio too short for segmentation.", ""
            # Pad if shorter than one segment but long enough to be meaningful
            audio_data_mono = np.pad(audio_data_mono, (0, seg_len_smp - len(audio_data_mono)), 'constant')

        max_start_point = len(audio_data_mono) - seg_len_smp
        if max_start_point < 0 : max_start_point = 0 # Should only happen if audio_data_mono became shorter than seg_len_smp after padding logic (unlikely)

        explanation_for_video = generate_simple_explanation(selected_feature_keys_for_explanation)

        for i in range(0, max_start_point + 1, step_smp):
            num_segs+=1
            seg_audio=audio_data_mono[i:i+seg_len_smp]
            if len(seg_audio) < cfg_analyzer.SAMPLE_RATE * 0.05 : continue # Skip very tiny residual segments

            input_tuple=extract_features_for_segment(seg_audio,cfg_analyzer.SAMPLE_RATE,norm_stats,cfg_analyzer.SELECTED_FEATURE_KEYS_OTHER)
            if input_tuple is None or not isinstance(input_tuple, tuple) or len(input_tuple) != 2:
                print(f"Warning: extract_features_for_segment returned invalid data for seg {num_segs}. Skipping.")
                continue
            if input_tuple[0] is None or input_tuple[0].nelement()==0:
                print(f"Warning: Spectrogram is None or empty for seg {num_segs}. Skipping.")
                continue

            spec_input, other_input = input_tuple
            if cfg_analyzer.OTHER_FEATURES_TOTAL_DIM_CURRENT > 0 and \
               (other_input is None or other_input.nelement() == 0 or (other_input.ndim > 1 and other_input.shape[1] == 0)):
                 other_input = torch.empty((spec_input.shape[0], 0, cfg_analyzer.MAX_LEN_OTHER_FEATURES_SEQ), dtype=torch.float32, device=device)

            with torch.no_grad():
                output=model((spec_input,other_input)); probs=nn.functional.softmax(output,dim=1)
                conf,pred_idx_t=torch.max(probs,1)

            pred_idx=pred_idx_t.item(); pred_code=str(pred_idx+1).zfill(2); pred_name=cfg_analyzer.EMOTIONS.get(pred_code,"Unk")
            conf_pct=conf.item()*100

            segment_result_data = {"segment_number": num_segs, "predicted_emotion": pred_name, "prediction_confidence_percent": conf_pct}
            list_of_segment_predictions.append(segment_result_data)

            if max_start_point == 0 and i == 0: break # If only one segment possible/processed

        overall_emotion_str = "N/A"
        if list_of_segment_predictions:
            hce = [p['predicted_emotion'] for p in list_of_segment_predictions if p['prediction_confidence_percent'] > 60.0]
            if hce: overall_emotion_str = Counter(hce).most_common(1)[0][0]
            else:
                all_e = [p['predicted_emotion'] for p in list_of_segment_predictions]
                if all_e: overall_emotion_str = Counter(all_e).most_common(1)[0][0]

        return list_of_segment_predictions, overall_emotion_str, explanation_for_video

    except Exception as e:
        print(f"ERROR: Exception during analysis of {current_video_path}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return [], f"Error: {e}", ""
    finally:
        # Ensure clips are closed and temp file is removed
        if audio_clip_mvp:
            try: audio_clip_mvp.close()
            except Exception as e_close: print(f"Warning: Error closing moviepy audio_clip: {e_close}")
        if video_clip:
            try: video_clip.close()
            except Exception as e_close: print(f"Warning: Error closing moviepy video_clip: {e_close}")
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except Exception as e_remove: print(f"Warning: Error removing temp audio file {temp_audio_path}: {e_remove}")

# ----------------- audio_tone class -----------------
class AudioTone(View):
    def __init__(self):
        super().__init__()
    
    def post(self, request):

        
        video_path = request.session.get('uploaded_video_path')
        if not video_path:
            print("CRITICAL ERROR: No video path found in session. Exiting.")
            return JsonResponse({"error": "No video path found."}, status=400)
        

        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_absolute_path):
            print(f"CRITICAL ERROR: Video file not found at {video_absolute_path}. Exiting.")
            return JsonResponse({"error": "Video file not found."}, status=404)
        

        cfg_analyzer.load_parameters_from_gg_config()
        set_seed(cfg_analyzer.SEED)


        if not os.path.exists(cfg_analyzer.MODEL_SAVE_PATH): print(f"CRITICAL ERROR: Model file NOT FOUND. Exiting."); sys.exit(1)
        if not os.path.exists(cfg_analyzer.NORM_STATS_SAVE_PATH): print(f"CRITICAL ERROR: Norm stats file NOT FOUND. Exiting."); sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Using device: {device}")

        if not cfg_analyzer.MODEL_ARCH_CONSTRUCTOR: print("CRITICAL ERROR: Model architecture not loaded. Exiting."); sys.exit(1)
        if cfg_analyzer.NUM_CLASSES == 0 or not cfg_analyzer.EMOTIONS : print("CRITICAL ERROR: Emotion classes/map not loaded. Exiting."); sys.exit(1)

        try:
            model_instance = DynamicAudioCRNN(
                spec_input_shape_hw=(cfg_analyzer.N_MELS, cfg_analyzer.MAX_LEN_SPECTROGRAM),
                current_other_features_dim=cfg_analyzer.OTHER_FEATURES_TOTAL_DIM_CURRENT,
                arch_params_constructor=cfg_analyzer.MODEL_ARCH_CONSTRUCTOR
            ).to(device)
            model_instance.load_state_dict(torch.load(cfg_analyzer.MODEL_SAVE_PATH, map_location=device))
            model_instance.eval()
        except Exception as e: print(f"CRITICAL ERROR: Failed to load model: {e}. Exiting."); sys.exit(1)

        try:
            norm_stats_loaded = torch.load(cfg_analyzer.NORM_STATS_SAVE_PATH, map_location=device)
            for k, v_dict in norm_stats_loaded.items():
                if isinstance(v_dict, dict):
                    for s_k in v_dict:
                        if isinstance(v_dict[s_k], torch.Tensor): v_dict[s_k] = v_dict[s_k].to(device)
                elif isinstance(v_dict, torch.Tensor): norm_stats_loaded[k] = v_dict.to(device)
        except Exception as e: print(f"CRITICAL ERROR: Failed to load normalization stats: {e}. Exiting."); sys.exit(1)

        print(f"\n--- Starting Video Analysis ---")
        

        segment_predictions, overall_emotion_for_video, explanation_for_video = analyze_video(
            video_absolute_path,
            model_instance,
            norm_stats_loaded,
            device,
            cfg_analyzer.SELECTED_FEATURE_KEYS_OTHER
        )

        
        if segment_predictions or (overall_emotion_for_video != "N/A" and not overall_emotion_for_video.startswith("Error:")) : # Check if results are valid
            print(f"  Overall Predicted Emotion: {overall_emotion_for_video}")
            print(f"  Model's Reasoning Basis: {explanation_for_video}")
            # Optionally print segment details if needed for debugging
            for seg_pred in segment_predictions:
                    print(f"    Segment {seg_pred['segment_number']}: {seg_pred['predicted_emotion']} ({seg_pred['prediction_confidence_percent']:.2f}%)")
        else:
            print(f"  ERROR analyzing : {overall_emotion_for_video}")

        print(f"\n\n{'='*20} ANALYSIS COMPLETE {'='*20}")
        print("Script finished processing the video.")


        return JsonResponse({
            "segment_predictions": segment_predictions,
            "overall_emotion": overall_emotion_for_video,
            "explanation": explanation_for_video
        })
'''

# ------------------ audio tone v2 class(rudra) ----------------
warnings.filterwarnings("ignore")

class AudioToneV2CRNN(nn.Module):
    """
    MODIFIED: This is the exact CRNN class from your 'run_full_pipeline.py' script.
    """
    def __init__(self, num_other_features, num_classes, arch_params):
        super(AudioToneV2CRNN, self).__init__()
        self.has_other_features = num_other_features > 0
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_output_features = 64 * 16
        lstm_input_size = self.cnn_output_features
        if self.has_other_features:
            lstm_input_size += num_other_features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=arch_params['lstm_hidden_size'],
            num_layers=arch_params['num_lstm_layers'],
            batch_first=True, bidirectional=True,
            dropout=arch_params['dropout'] if arch_params['num_lstm_layers'] > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(arch_params['lstm_hidden_size'] * 2, arch_params['fc_hidden_size']),
            nn.ReLU(),
            nn.Dropout(arch_params['dropout']),
            nn.Linear(arch_params['fc_hidden_size'], num_classes)
        )

    def forward(self, spectrogram, other_features):
        cnn_out = self.cnn(spectrogram)
        batch_size, _, _, time_steps = cnn_out.size()
        cnn_out = cnn_out.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)
        if self.has_other_features and other_features.numel() > 0:
            other_features = other_features.permute(0, 2, 1)
            other_features_aligned = other_features[:, :time_steps, :]
            combined_features = torch.cat((cnn_out, other_features_aligned), dim=2)
        else:
            combined_features = cnn_out
        lstm_out, _ = self.lstm(combined_features)
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_time_step_out)
        return logits

class AudioTone_V2(View):
    def __init__(self):
        # Automatically determine file paths from the run directory
        self.CONFIG_YAML_PATH = "/home/student/new_api/testpro-main/models/audio_tone/rudra/final_run_config.yaml"
        self.MODEL_LOAD_PATH = "/home/student/new_api/testpro-main/models/audio_tone/rudra/final_model_best.pth"
        self.SAMPLE_RATE = 16000
        self.N_MELS = 128
        self.MAX_LEN_FRAMES = 250
        self.OTHER_FEATURE_DIMS = {'mfcc': 20, 'rms': 1, 'chroma': 12}
        self.BEST_FEATURES = []
        self.SEGMENT_DURATION_S = 4.0
        self.SEGMENT_OVERLAP_S = 2.0
        self.EMOTIONS = {}
        self.EMOTION_MAP = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5}
        self.SEED = 42
        self.NUM_CLASSES = 0
        self.MODEL_ARCHITECTURE = {}


    def load_parameters_from_config(self):
        """
        MODIFIED: Loads parameters from the YAML created by run_full_pipeline.py.
        """
        if not os.path.exists(self.CONFIG_YAML_PATH):
            print(f"CRITICAL ERROR: Config YAML NOT FOUND: {self.CONFIG_YAML_PATH}")
            sys.exit(1)
        try:
            with open(self.CONFIG_YAML_PATH, 'r') as f:
                loaded_cfg = yaml.safe_load(f)
            print(f"INFO: Successfully loaded config from: {self.CONFIG_YAML_PATH}")

            # Load the correct keys
            self.MODEL_ARCHITECTURE = loaded_cfg['best_architecture']
            self.BEST_FEATURES = loaded_cfg['best_features']
            self.EMOTIONS = self.EMOTION_MAP # Use the hardcoded map as it's not saved in the new yaml
            self.NUM_CLASSES = len(self.EMOTIONS)

            print(f"  Config Loaded: Model Architecture = {self.MODEL_ARCHITECTURE}")
            print(f"  Config Loaded: Best Features = {self.BEST_FEATURES}")
            return True

        except KeyError as ke:
            print(f"CRITICAL ERROR: Key {ke} missing in YAML. The YAML file and this script are out of sync.")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"CRITICAL ERROR: Problem loading/parsing config YAML: {e}")
            traceback.print_exc()
            sys.exit(1)

    def extract_features_for_segment(self,waveform_np):
        """
        MODIFIED: Simplified feature extraction to match the training pipeline exactly.
        No scaler or separate normalization stats are needed.
        """
        try:
            # Helper to pad/truncate
            def _pad(array, max_len):
                if array.shape[1] > max_len: return array[:, :max_len]
                padding = np.zeros((array.shape[0], max_len - array.shape[1]))
                return np.hstack((array, padding))

            # Spectrogram
            spectrogram = librosa.feature.melspectrogram(y=waveform_np, sr=self.SAMPLE_RATE, n_mels=self.N_MELS)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            padded_spectrogram = _pad(log_spectrogram, self.MAX_LEN_FRAMES)
            spec_tensor = torch.from_numpy(padded_spectrogram).float().unsqueeze(0) # Add channel dim

            # Other features (MFCC, RMS, Chroma)
            mfcc = librosa.feature.mfcc(y=waveform_np, sr=self.SAMPLE_RATE, n_mfcc=20)
            rms = librosa.feature.rms(y=waveform_np)
            chroma = librosa.feature.chroma_stft(y=waveform_np, sr=self.SAMPLE_RATE)
            other_features_unpadded = np.vstack((mfcc, rms, chroma))
            padded_other_features = _pad(other_features_unpadded, self.MAX_LEN_FRAMES)

            # Select the specific features the model was trained on
            rows_to_select = []
            current_pos = 0
            for key in ['mfcc', 'rms', 'chroma']:
                dim = self.OTHER_FEATURE_DIMS[key]
                if key in self.BEST_FEATURES:
                    rows_to_select.extend(range(current_pos, current_pos + dim))
                current_pos += dim

            selected_features = padded_other_features[rows_to_select, :]
            other_tensor = torch.from_numpy(selected_features).float()

            return spec_tensor, other_tensor

        except Exception as e:
            print(f"  WARN: Feature extraction failed for a segment: {e}")
            return None, None

    def analyze_video(self,current_video_path, model_to_use, device_to_use):
        """
        MODIFIED: This version uses a more robust method for audio extraction to
                avoid the common moviepy 'stacking' TypeError.
        """
        video_basename = os.path.basename(current_video_path)
        if not os.path.exists(current_video_path):
            print(f"ERROR: Video file not found: {current_video_path}")
            return [], f"Video file '{video_basename}' not found."

        # NEW: Define a temporary file path for the audio
        temp_audio_path = "temp_audio_for_analysis.wav"
        
        try:
            print(f"  Processing video: {video_basename}...")
            with VideoFileClip(current_video_path) as video_clip:
                if video_clip.audio is None:
                    return [], "No audio track."

                # STEP 1: Write audio to a temporary .wav file. This is more robust.
                # The 'pcm_s16le' codec creates a standard uncompressed WAV file.
                video_clip.audio.write_audiofile(
                    temp_audio_path,
                    fps=self.SAMPLE_RATE,
                    codec='pcm_s16le',
                    verbose=False, # Suppress moviepy's console output
                    logger=None    # Suppress moviepy's console output
                )

            # STEP 2: Load the audio from the temporary file using librosa.
            # librosa.load is very reliable and handles conversion to mono automatically.
            audio_data_mono, _ = librosa.load(temp_audio_path, sr=self.SAMPLE_RATE, mono=True)
            audio_data_mono = audio_data_mono.astype(np.float32)

            # --- The rest of the logic remains the same ---
            segment_len = int(self.SEGMENT_DURATION_S * self.SAMPLE_RATE)
            step = int((self.SEGMENT_DURATION_S - self.SEGMENT_OVERLAP_S) * self.SAMPLE_RATE)

            predictions = []
            if len(audio_data_mono) < segment_len:
                print(f"  WARN: Audio track is shorter than one segment ({self.SEGMENT_DURATION_S}s). Padding to analyze.")
                audio_data_mono = np.pad(audio_data_mono, (0, segment_len - len(audio_data_mono)), 'constant')


            for i, start in enumerate(range(0, len(audio_data_mono) - segment_len + 1, step)):
                segment = audio_data_mono[start : start + segment_len]

                spec_tensor, other_tensor = self.extract_features_for_segment(segment)
                if spec_tensor is None:
                    continue

                spec_tensor = spec_tensor.unsqueeze(0).to(device_to_use)
                other_tensor = other_tensor.unsqueeze(0).to(device_to_use)

                with torch.no_grad():
                    logits = model_to_use(spec_tensor, other_tensor)

                probs = nn.functional.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                emotion_name = [name for name, idx in self.EMOTIONS.items() if idx == pred_idx.item()][0]

                predictions.append({"segment": i + 1, "emotion": emotion_name, "confidence": confidence.item()})

            if not predictions:
                return [], "No valid segments to analyze."

            emotion_counts = Counter(p['emotion'] for p in predictions)
            overall_emotion = emotion_counts.most_common(1)[0][0]

            return predictions, overall_emotion

        except Exception as e:
            print(f"ERROR: Exception during analysis of '{video_basename}': {e}")
            traceback.print_exc()
            return [], f"RuntimeError: {e}"
        
        finally:
            # STEP 3: Clean up the temporary file, no matter what happens.
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


    def post(self,request):
        if not self.load_parameters_from_config():
            sys.exit(1) # Exit if config loading fails

        random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Analysis device: {device.type.upper()}")

        if not os.path.exists(self.MODEL_LOAD_PATH):
            print(f"CRITICAL ERROR: Model file NOT FOUND: {self.MODEL_LOAD_PATH}")
            sys.exit(1)

        try:
            # Calculate the number of input features for the 'other_features' stream
            num_other_features = sum(self.OTHER_FEATURE_DIMS[key] for key in self.BEST_FEATURES)

            # Instantiate the correct model class with parameters from the config
            model = AudioToneV2CRNN(
                num_other_features=num_other_features,
                num_classes=self.NUM_CLASSES,
                arch_params=self.MODEL_ARCHITECTURE
            ).to(device)
            
            # Load the saved weights from the .pth file
            model.load_state_dict(torch.load(self.MODEL_LOAD_PATH, map_location=device))
            model.eval() # Set model to evaluation mode
            print(f"INFO: Model loaded successfully from {self.MODEL_LOAD_PATH}")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load the model: {e}")
            traceback.print_exc()
            sys.exit(1)

        video_path = request.session.get('uploaded_video_path')
        if not video_path:
            print("CRITICAL ERROR: No video path found in session. Exiting.")
            return JsonResponse({"error": "No video path found."}, status=400)
        

        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_absolute_path):
            print(f"CRITICAL ERROR: Video file not found at {video_absolute_path}. Exiting.")
            return JsonResponse({"error": "Video file not found."}, status=404)
        

        if not os.path.exists(video_absolute_path):
            print(f"WARNING: Video file NOT FOUND: {video_absolute_path}. Skipping.")
        else:
            segment_preds, overall_emotion = self.analyze_video(video_absolute_path, model, device)

            print(f"\n--- Results for: {os.path.basename(video_absolute_path)} ---")
            if segment_preds:
                print(f"  Overall Predicted Emotion: {overall_emotion}")
                print("  Segment Details:")
                for seg_data in segment_preds:
                    print(f"    - Segment {seg_data['segment']}: {seg_data['emotion']} ({seg_data['confidence']:.1%})")
            else:
                print(f"  Analysis FAILED: {overall_emotion}")

        print(f"\n--- ANALYSIS COMPLETE ---")

        return JsonResponse({
            "overall_emotion": overall_emotion,
            "segment_predictions": segment_preds,
        })

# ----------------- audio tone v3 class(souradeep)
warnings.filterwarnings('ignore')

class AudioToneV3CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,4))
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 130)
            cnn_out = self.cnn(dummy_input)
            self.gru_input_size = cnn_out.shape[1] * cnn_out.shape[2]
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=256, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(256 * 2, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, _, _ = x.shape
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :] 
        x = self.fc(x)
        return x

class AudioTone_V3(View):
    def __init__(self):
        self.MODEL_PATH = "/home/student/new_api/testpro-main/models/audio_tone/souradeep/definitive_crnn_model.pth"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAMPLE_RATE = 22050
        self.FIXED_DURATION_S = 3
        self.N_MELS = 128
        self.HOP_LENGTH = 512

        self.CLASS_NAMES = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']
        self.N_CLASSES = len(self.CLASS_NAMES)


    def preprocess_audio(self,audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            target_length = int(self.FIXED_DURATION_S * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            else:
                audio = audio[:target_length]
                
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.N_MELS, hop_length=self.HOP_LENGTH)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            spec_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return spec_tensor
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return None

    def plot_predictions(self,probabilities, video_filename):
        y_pos = np.arange(len(self.CLASS_NAMES))
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(y_pos, probabilities, align='center', color='skyblue')
        plt.yticks(y_pos, self.CLASS_NAMES)
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel('Confidence')
        plt.title(f'Emotion Prediction Confidence for "{os.path.basename(video_filename)}"')
        plt.gca().invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', va='center')
            
        plt.xlim(0, 1.1)
        plt.tight_layout()
        plt.show()

    def analysis(self,video_path):
        print("Loading the definitive CRNN model...")
        try:
            model = AudioToneV3CRNN(n_classes=self.N_CLASSES)
            model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.DEVICE, weights_only=True))
            model.to(self.DEVICE)
            model.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at '{self.MODEL_PATH}'.")
            print("Please make sure you have successfully run the training script.")
            exit()

        print("\n" + "="*60)

        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found: {video_path}")

        print(f"Analyzing: {os.path.basename(video_path)}")
        
        temp_audio_path = "temp_audio.wav"
        
        try:
            print("Step 1: Extracting audio...")
            video_clip = VideoFileClip(video_path)
            
            # --- THE FIX IS HERE ---
            video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            # --- END OF FIX ---
            
            video_clip.close()

            print("Step 2: Preprocessing audio into a spectrogram...")
            spec_tensor = self.preprocess_audio(temp_audio_path)
            
            if spec_tensor is None:
                pass

            print("Step 3: Predicting emotion...")
            with torch.no_grad():
                spec_tensor = spec_tensor.to(self.DEVICE)
                logits = model(spec_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                top_prob, top_idx = torch.max(probabilities, dim=0)
                predicted_emotion = self.CLASS_NAMES[top_idx]
            
            print("\n--- PREDICTION RESULT ---")
            print(f"The model predicts the dominant emotion is: {predicted_emotion.upper()}")
            print(f"Confidence: {top_prob.item():.2%}")
            print("-------------------------")
            
            self.plot_predictions(probabilities.cpu().numpy(), video_path)

        except Exception as e:
            print(f"An unexpected error occurred while processing {video_path}: {e}")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
                    
        print("\nAnalysis complete.")

        return predicted_emotion,round(top_prob.item(),4)*100
    
    def post(self,request):
        video_path = request.session.get('uploaded_video_path')
        if not video_path:
            print("CRITICAL ERROR: No video path found in session. Exiting.")
            return JsonResponse({"error": "No video path found."}, status=400)
        
        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_absolute_path):
            print(f"CRITICAL ERROR: Video file not found at {video_absolute_path}. Exiting.")
            return JsonResponse({"error": "Video file not found."}, status=404)
        

        if not os.path.exists(video_absolute_path):
            print(f"WARNING: Video file NOT FOUND: {video_absolute_path}. Skipping.")
        else:
            pred_emotion,confidence = self.analysis(video_absolute_path)

        print(f"\n--- ANALYSIS COMPLETE ---")

        return JsonResponse({
            "predicted_emotion": pred_emotion,
            "confidence": confidence,
        })



###################### audio tone end ##########################

###################### HeartRate class ######################### 
  
class HeartRate(View):

    def __init__(self):
        super().__init__()

        model_path = "/home/student/new_api/testpro-main/models/rppgnet_hr_prediction_50.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            self.model = None
        else:
            try:
                self.model = tf.keras.models.load_model(model_path,
                                                    custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
                print("TensorFlow Heart Rate model loaded successfully.")
            except Exception as e:
                print(f"Error loading TensorFlow Heart Rate model: {e}")
                self.model = None


        self.FRAMES_PER_SEQUENCE = 30
        self.FRAME_SIZE = (64, 64)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar cascade XML file!")
        else:
            print("Haar cascade face detector loaded successfully.")



    def extract_face_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            x, y, w, h = max(0, x), max(0, y), min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                return cv2.resize(face_roi, self.FRAME_SIZE), (x, y, w, h)

        return None, None

    def estimate_emotion(self, hr):

        if hr < 60:
            return "Calm / Relaxed", (0, 255, 0)
        elif 60 <= hr < 80:
            return "Neutral / Normal", (0, 255, 255)
        elif 80 <= hr < 100:
            return "Moderate Excitement", (255, 0, 0)
        elif 100 <= hr < 140:
            return "High Excitement / Stress", (128, 0, 128)
        else:
             return "Extreme Arousal", (0, 0, 255)


    def calculate_hr_trust_score(self, hr):
        if hr is None:
            return 0.0

        if hr < 40 or hr > 180:
            return 20.0
        elif (hr >= 40 and hr < 50) or (hr > 140 and hr <= 180):
            return 50.0
        elif (hr >= 50 and hr < 60) or (hr > 120 and hr <= 140):
            return 70.0
        elif (hr >= 60 and hr < 90):
            return 95.0
        elif (hr >= 90 and hr <= 120):
            return 90.0
        else:
            return 50.0


    def calculate_overall_hr_metrics(self, heart_rates, emotion_list):
        if not heart_rates:
            return 0.0, 0.0, "Undetermined"

        avg_hr = np.mean(heart_rates)

        hr_std = np.std(heart_rates)
        variability_trust = max(0, 100 - hr_std * 5)

        normal_range_count = sum(1 for hr in heart_rates if 60 <= hr <= 120)
        normal_range_trust = (normal_range_count / len(heart_rates)) * 100 if heart_rates else 0

        overall_trust_factor = (variability_trust * 0.5 + normal_range_trust * 0.5)

        emotion_counts = Counter(emotion_list)
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_list else "Undetermined"

        return round(avg_hr, 1), round(overall_trust_factor, 1), dominant_emotion


    def process_video(self, video_path):
    
        
        if self.model is None:
            return "Error: Heart Rate model not loaded.", 0.0, "Undetermined", None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Cannot open video file", 0.0, "Undetermined", None


        frames_sequence = []
        heart_rates_history = []
        emotion_history = []

        current_hr_display = "--"
        current_emotion_display = "Analyzing..."
        current_emotion_color = (128, 128, 128)


        frame_count = 0  # Initialize frame counter
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            annotated_frame = frame.copy()

            face_roi, bbox = self.extract_face_roi(annotated_frame)

            if face_roi is not None:
                frames_sequence.append(face_roi)
                x, y, w, h = bbox
                

            if len(frames_sequence) == self.FRAMES_PER_SEQUENCE:
                if self.model:
                    try:
                        input_data = np.array(frames_sequence, dtype=np.float32) / 255.0
                        input_data = np.expand_dims(input_data, axis=0)

                        hr_prediction = self.model.predict(input_data, verbose=0)[0][0]
                        hr_value = int(hr_prediction)

                        heart_rates_history.append(hr_value)
                        if len(heart_rates_history) > 300:
                             heart_rates_history.pop(0)

                        current_hr_display = f"{hr_value}"
                        emotion, current_emotion_color = self.estimate_emotion(hr_value)
                        emotion_history.append(emotion)
                        if len(emotion_history) > 300:
                            emotion_history.pop(0)
                        current_emotion_display = emotion


                    except Exception as e:
                        print(f"Error during HR prediction: {e}")
                        current_hr_display = "Error"
                        current_emotion_display = "Error"
                        current_emotion_color = (0, 0, 255)

                frames_sequence = []
            frame_count += 1  # Increment frame counter

        cap.release()

        overall_avg_hr, overall_trust_factor, overall_dominant_emotion = self.calculate_overall_hr_metrics(heart_rates_history, emotion_history)

        
        # Return report filename along with other results
        return overall_avg_hr, overall_trust_factor, overall_dominant_emotion


    def post(self, request):
        if self.model is None:
            return JsonResponse({"error": "Heart rate prediction model failed to load."}, status=500)
        if self.face_cascade.empty():
            return JsonResponse({"error": "Face detection model failed to load."}, status=500)

        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            output_dir = os.path.join(settings.MEDIA_ROOT, "output_videos")
            os.makedirs(output_dir, exist_ok=True)

            overall_avg_hr, overall_trust_factor, overall_dominant_emotion = self.process_video(video_absolute_path)

            if isinstance(overall_avg_hr, str):
                return JsonResponse({"error": overall_avg_hr, "status": "error"}, status=500)


            content = {
                
                'avg_hr': float(overall_avg_hr),
                
                'dominant_emotion': overall_dominant_emotion,
                
                'status': 'success'
            }

            return JsonResponse(content)

        except FileNotFoundError as e:
            print(f"File error in heart rate analysis: {str(e)}")
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except IOError as e:
             print(f"IO error in heart rate analysis: {str(e)}")
             return JsonResponse({
                 "error": "An error occurred while loading a required resource.",
                 "details": str(e),
                 "status": "error"
             }, status=500)
        except Exception as e:
            print(f"Error in heart rate analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)
        
# ----------------- HRV class -----------------

'''
class HRV(View):
    def __init__(self):
        super().__init__()
        # Load Pre-trained RPPGNet IBI Prediction Model
        model_path = "/home/student/new_api/testpro-main/models/rppgnet_ibi_prediction.h5"
        spo2_model_path = "/home/student/new_api/testpro-main/models/rPPG_spo2_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: HRV model file not found at {model_path}")
            self.model = None
        else:
            try:
                self.model = tf.keras.models.load_model(model_path,
                                                        custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

            except Exception as e:
                print(f"Error loading HRV model: {e}")
                self.model = None
        if not os.path.exists(spo2_model_path):
            print(f"Error: SPO2 model file not found at {spo2_model_path}")
            self.spo2_model = None
        else:
            try:
                self.spo2_model = tf.keras.models.load_model(spo2_model_path,
                                                             compile=False)
            except Exception as e:
                print(f"Error loading SPO2 model: {e}")


        # Load Haar Cascade for face detection
        haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_cascade_path):
            print(f"Error: Haar cascade file not found at {haar_cascade_path}")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_cascade.empty():
                print(f"Error: Failed to load Haar cascade XML file from {haar_cascade_path}")
                self.face_cascade = None

        self.FRAMES_PER_SEQUENCE = 30
        self.FRAME_SIZE = (64, 64)
        self.IBI_VALUES_PER_SEQUENCE = 10
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        self.GRAPH_PADDING = 15
        self.BLUR_KERNEL_BG = (15, 15)
        self.BLUR_SIGMA_BG = 10
        self.BLUR_KERNEL_OVERLAY = (49, 49)
        self.BLUR_SIGMA_OVERLAY = 15
        self.BLEND_WEIGHT_BG = 0.4
        self.BLEND_WEIGHT_OVERLAY = 0.6

        self.squence_length = 30  # Number of frames in each sequence
        self.frame_skip = 1
        self.roi_size = 64  # Size of the face ROI for processing
        self.EMOTION_COLORS = {
            "Happy": "green",
            "Sad": "brown",
            "Fearful": "purple",
            "Stressed": "red",
            "Anxious": "orange",
            "Neutral": "blue",
            "Excited": "cyan"
        }


    def extract_face_roi(self, frame):
        if self.face_cascade is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x, y, w, h = max(0, x), max(0, y), min(frame.shape[1]-x, w), min(frame.shape[0]-y, h)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                return cv2.resize(face, self.FRAME_SIZE), (x, y, w, h)
        return None, None

    def extract_face_roi_spo2(self, image, roi_size=64):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            if face.size > 0:
                return cv2.resize(face, (roi_size, roi_size))
        return None


    def compute_rmssd(self, ibi_sequence):
        ibi_sequence = np.asarray(ibi_sequence)
        if len(ibi_sequence) < 2:
            return 0.0
        diff = np.diff(ibi_sequence)
        return np.sqrt(np.mean(diff ** 2))

    def compute_hr(self, ibi_sequence_ms):
        ibi_sequence_ms = np.asarray(ibi_sequence_ms)
        if len(ibi_sequence_ms) == 0 or np.mean(ibi_sequence_ms) == 0:
            return 0.0
        return 60000.0 / np.mean(ibi_sequence_ms)

    def compute_sdnn(self, all_ibi_values):
        all_ibi_values = np.asarray(all_ibi_values)
        if len(all_ibi_values) < 2:
            return 0.0
        return np.std(all_ibi_values)

    def estimate_emotion(self, hr, hrv):
        hr = float(hr)
        hrv = float(hrv)

        if hr > 120 and hrv < 20:
                 return "Fearful"
        elif hr > 100 and hrv > 50:
            return "Excited"
        elif hr < 70 and hrv > 50:
            return "Happy"
        elif hr > 100 and hrv < 30:
            return "Stressed"
        elif hr > 90 and hrv < 30:
            return "Anxious"
        elif hr < 70 and hrv < 30:
            return "Sad"
        else:
            return "Neutral"

    # Emotion classification based on SpO2
    def classify_emotion(self, spo2_value):
        if spo2_value is None:
             return "Analyzing..."
        if spo2_value > 97:
            return "Calm / Relaxed"
        elif 95 < spo2_value <= 97:
            return "Neutral / Normal"
        elif 92 < spo2_value <= 95:
            return "Moderate Excitement"
        elif 90 < spo2_value <= 92:
            return "High Excitement / Stress"
        elif spo2_value <= 90:
            return "Extreme Arousal / Panic"
        else:
            return "Uncertain / Mixed State"


    def predict_ibi_from_video(self, video_path):
        
        if self.model is None or self.face_cascade is None:
            return 0, 0, "Error: Model or face cascade not loaded.", 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0, "Error: Cannot open video file", 0

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        roi_buffer = []
        frame_buffer = []
        spo2_predictions = []
        emotion_predictions = []
        hr_list, hrv_list, all_ibi_values = [], [], []
        current_hr_to_display = 0.0
        current_hrv_to_display = 0.0
        current_emotion_to_display = "Analyzing..."
        current_spo2_to_display = None # Initialize as None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            annotated_frame = frame.copy()

            face_roi, face_coords = self.extract_face_roi(annotated_frame)

            # Process for SpO2 prediction
            spo2_roi = self.extract_face_roi_spo2(original_frame, self.roi_size)
            if spo2_roi is not None:
                frame_buffer.append(spo2_roi)
            if len(frame_buffer) > self.squence_length:
                frame_buffer.pop(0)  # Keep only the last N frames

            if len(frame_buffer) == self.squence_length and self.spo2_model is not None:
                try:
                    seq = np.array(frame_buffer).astype(np.float32) / 255.0
                    seq = np.expand_dims(seq, axis=0)
                    pred_spo2 = self.spo2_model.predict(seq, verbose=0)[0][0]
                    spo2_predictions.append(pred_spo2)

                    current_spo2_to_display = pred_spo2

                    emotion = self.classify_emotion(pred_spo2)
                    emotion_predictions.append(emotion)

                    current_emotion_to_display = emotion
                except Exception as e:
                     print(f"Error predicting SpO2 or emotion: {e}")
                     current_spo2_to_display = None # Reset on error
                     current_emotion_to_display = "Error"
                     pass


            # Process for IBI prediction (HRV/HR)
            if face_coords is not None:
                x, y, w, h = face_coords
                

                if face_roi is not None:
                    roi_buffer.append(face_roi / 255.0)

            if len(roi_buffer) == self.FRAMES_PER_SEQUENCE and self.model is not None:
                try:
                    input_sequence = np.expand_dims(np.array(roi_buffer), axis=0)
                    predicted_ibi_sequence = self.model.predict(input_sequence, verbose=0)[0] * 1000

                    hr_seq = self.compute_hr(predicted_ibi_sequence)
                    hrv_seq = self.compute_rmssd(predicted_ibi_sequence)

                    roi_buffer = [] # Clear buffer after prediction

                    hr_list.append(hr_seq)
                    hrv_list.append(hrv_seq)
                    all_ibi_values.extend(predicted_ibi_sequence)


                    current_hr_to_display = hr_seq
                    current_hrv_to_display = hrv_seq

                except Exception as e:
                    print(f"Error predicting IBI/HR/HRV: {e}")
                    current_hr_to_display = 0.0
                    current_hrv_to_display = 0.0
                    pass

            frame_idx += 1

        cap.release()

        final_mean_hr = np.mean(hr_list) if hr_list else 0.0
        final_sdnn = self.compute_sdnn(all_ibi_values)
        average_spo2 = np.mean(spo2_predictions) if spo2_predictions else 0.0

        most_common_emotion = "N/A"
        if emotion_predictions:
            emotion_counter = Counter(emotion_predictions)
            most_common_emotion, count = emotion_counter.most_common(1)[0]
            print(f"Most common emotion: {most_common_emotion} with count: {count}")
    
        return final_sdnn, final_mean_hr, most_common_emotion, average_spo2


    def post(self, request):
        if self.model is None or self.face_cascade is None or self.spo2_model is None:
            error_message = "Analysis models not loaded. "
            if self.model is None: error_message += "HRV model missing. "
            if self.face_cascade is None: error_message += "Face cascade missing. "
            if self.spo2_model is None: error_message += "SpO2 model missing. "
            return JsonResponse({"error": error_message.strip()}, status=500)


        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)


            sdnn, mean_hr, overall_emotion, average_spo2= self.predict_ibi_from_video(video_absolute_path)

            if "Error" in overall_emotion or "Error" in str(sdnn):
                 return JsonResponse({"error": f"Analysis failed: {overall_emotion}" if "Error" in overall_emotion else "Analysis failed.",
                                       "details": f"SDNN: {sdnn}, Mean HR: {mean_hr}, Avg SpO2: {average_spo2}"}, status=500)

            content = {
            
                'sdnn': float(sdnn),
                'mean_hr': float(mean_hr),
                'status': 'success',
                'avg_emotion': overall_emotion,
                'average_spo2': float(average_spo2),
                
            }

            return JsonResponse(content)

        except FileNotFoundError as e:
            print(f"File error in HRV analysis: {str(e)}")
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            print(f"Error in HRV analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)

    def generate_emotion_reason(self, mean_hr, sdnn, avg_emotion):
        reason = f"Based on a mean HR of {mean_hr:.1f} bpm and SDNN of {sdnn:.1f} ms, "
        if avg_emotion == "Excited":
            return reason + "a high heart rate and high variability indicate strong positive arousal."
        elif avg_emotion == "Happy":
            return reason + "a low heart rate and high variability suggest a calm and positive emotional state."
        elif avg_emotion == "Stressed":
            return reason + "a high heart rate with low variability is often linked to stress."
        elif avg_emotion == "Anxious":
            return reason + "elevated heart rate and reduced variability commonly reflect anxiety."
        elif avg_emotion == "Neutral":
            return reason + "heart rate and variability lie within typical resting range, implying a neutral state."
        elif avg_emotion == "Sad":
            return reason + "a low heart rate and low variability suggest a subdued or sad emotional condition."
        elif avg_emotion == "Fearful":
            return reason + "a very high heart rate and extremely low variability are associated with fear."
        else:
            return reason + "the metrics suggest a balanced physiological state."
'''


#----------------- HRV v2 class -----------------------
HR_ZONES = {
    "Low (<60)": "#1f77b4",  # Blue
    "Normal (60-100)": "#2ca02c",  # Green
    "High (>100)": "#d62728"  # Red
}

class HRV(View):
    def __init__(self):
        super().__init__()
        # Load Pre-trained RPPGNet IBI Prediction Model
        model_path = "/home/student/new_api/testpro-main/models/rppgnet_hr_prediction_50.h5"
        spo2_model_path = "/home/student/new_api/testpro-main/models/rPPG_spo2_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: HRV model file not found at {model_path}")
            self.model = None
        else:
            try:
                self.model = tf.keras.models.load_model(model_path,
                                                        custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

            except Exception as e:
                print(f"Error loading HRV model: {e}")
                self.model = None
        if not os.path.exists(spo2_model_path):
            print(f"Error: SPO2 model file not found at {spo2_model_path}")
            self.spo2_model = None
        else:
            try:
                self.spo2_model = tf.keras.models.load_model(spo2_model_path,
                                                             compile=False)
            except Exception as e:
                print(f"Error loading SPO2 model: {e}")


        # Load Haar Cascade for face detection
        haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_cascade_path):
            print(f"Error: Haar cascade file not found at {haar_cascade_path}")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_cascade.empty():
                print(f"Error: Failed to load Haar cascade XML file from {haar_cascade_path}")
                self.face_cascade = None

        self.FRAMES_PER_SEQUENCE = 30
        self.FRAME_SIZE = (64, 64)
        self.IBI_VALUES_PER_SEQUENCE = 10
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        self.GRAPH_PADDING = 15
        self.BLUR_KERNEL_BG = (15, 15)
        self.BLUR_SIGMA_BG = 10
        self.BLUR_KERNEL_OVERLAY = (49, 49)
        self.BLUR_SIGMA_OVERLAY = 15
        self.BLEND_WEIGHT_BG = 0.4
        self.BLEND_WEIGHT_OVERLAY = 0.6

        self.squence_length = 30  # Number of frames in each sequence
        self.frame_skip = 1
        self.roi_size = 64  # Size of the face ROI for processing
        self.avg_heart_rate = 0.0
        self.avg_respiration_rate = 0.0
        self.detrended_hr = []
        self.frequencies = []
        self.heart_rate_variability = 0.0
        self.heart_rates = []
        self.hrv_sdnn = 0.0
        self.ibi = []
        self.mask = None
        self.max_heart_rate = 0.0
        self.min_heart_rate = 0.0
        self.n = 0  # Sample size or sequence length
        self.power = []
        self.resp_rate_hz = 0.0
        self.EMOTION_COLORS = {
            "Happy": "green",
            "Sad": "brown",
            "Fearful": "purple",
            "Stressed": "red",
            "Anxious": "orange",
            "Neutral": "blue",
            "Excited": "cyan"
        }


    def extract_face_roi(self, frame):
        if self.face_cascade is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x, y, w, h = max(0, x), max(0, y), min(frame.shape[1]-x, w), min(frame.shape[0]-y, h)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                return cv2.resize(face, self.FRAME_SIZE), (x, y, w, h)
        return None, None

    def extract_face_roi_spo2(self, image, roi_size=64):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            if face.size > 0:
                return cv2.resize(face, (roi_size, roi_size))
        return None


    def compute_rmssd(self, ibi_sequence):
        ibi_sequence = np.asarray(ibi_sequence)
        if len(ibi_sequence) < 2:
            return 0.0
        diff = np.diff(ibi_sequence)
        return np.sqrt(np.mean(diff ** 2))

    def compute_hr(self, ibi_sequence_ms):
        ibi_sequence_ms = np.asarray(ibi_sequence_ms)
        if len(ibi_sequence_ms) == 0 or np.mean(ibi_sequence_ms) == 0:
            return {
                "avg_heart_rate": 0.0,
                "max_heart_rate": 0.0,
                "min_heart_rate": 0.0,
                "heart_rates": []
            }
        heart_rates = 60000.0 / ibi_sequence_ms
        self.avg_heart_rate = np.mean(heart_rates)
        self.max_heart_rate = np.max(heart_rates)
        self.min_heart_rate = np.min(heart_rates)
        self.heart_rates = heart_rates.tolist()
        return {
            "avg_heart_rate": self.avg_heart_rate,
            "max_heart_rate": self.max_heart_rate,
            "min_heart_rate": self.min_heart_rate,
            "heart_rates": self.heart_rates
        }
    
    def compute_sdnn(self, all_ibi_values):
        all_ibi_values = np.asarray(all_ibi_values)
        if len(all_ibi_values) < 2:
            return 0.0
        return np.std(all_ibi_values)


    def compute_avg_respiration_rate(self, ibi_sequence):
        if len(ibi_sequence) < 4:
            return 0.0
        heart_rates = np.array([60000.0 / ibi for ibi in ibi_sequence])
        detrended_hr = heart_rates - np.mean(heart_rates)
        n = len(detrended_hr)
        freqs = np.fft.rfftfreq(n, d=1.0)
        power_spectrum = np.abs(np.fft.rfft(detrended_hr)) ** 2
        # Limit to typical respiration frequency range: 0.1 - 0.5 Hz
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if np.any(mask):
            peak_freq = freqs[mask][np.argmax(power_spectrum[mask])]
            return peak_freq * 60  # Convert Hz to breaths/min
        return 0.0

    def calculate_scaled_dimensions(self, frame_width, frame_height, target_width, target_height):
        aspect_ratio = frame_width / frame_height
        target_aspect = target_width / target_height

        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            y_offset = (target_height - new_height) // 2
            return new_width, new_height, 0, y_offset
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            x_offset = (target_width - new_width) // 2
            return new_width, new_height, x_offset, 0

    def prepare_watermark_hrv(self, watermark_path, target_width, target_height):
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        if watermark is None:
            print(f"Warning: Watermark file not found at {watermark_path}")
            return None

        if watermark.shape[2] == 3:
            alpha = np.ones((watermark.shape[0], watermark.shape[1], 1), dtype=watermark.dtype) * 255
            watermark = cv2.merge([watermark, alpha])

        diagonal_length = int(np.sqrt(target_width*2 + target_height*2) * 0.4)
        aspect_ratio = watermark.shape[1] / watermark.shape[0]
        new_width = diagonal_length
        new_height = int(new_width / aspect_ratio)
        if new_height <= 0: new_height = 1

        try:
            watermark_resized = cv2.resize(watermark, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        except cv2.error as e:
            print(f"Error resizing watermark: {e}")
            return None

        pad_size = max(new_width, new_height)
        padded = np.zeros((pad_size, pad_size, 4), dtype=np.uint8)
        x_offset = (pad_size - new_width) // 2
        y_offset = (pad_size - new_height) // 2

        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = watermark_resized

        angle = np.degrees(np.arctan2(target_height, target_width))
        matrix = cv2.getRotationMatrix2D((pad_size/2, pad_size/2), angle, 1.0)
        rotated = cv2.warpAffine(padded, matrix, (pad_size, pad_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        final_watermark = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        x_center_rotated = rotated.shape[1] // 2
        y_center_rotated = rotated.shape[0] // 2

        crop_x = x_center_rotated - target_width // 2
        crop_y = y_center_rotated - target_height // 2

        crop_x_start = max(0, crop_x)
        crop_y_start = max(0, crop_y)
        crop_x_end = min(rotated.shape[1], crop_x + target_width)
        crop_y_end = min(rotated.shape[0], crop_y + target_height)

        final_x_start = max(0, -crop_x)
        final_y_start = max(0, -crop_y)

        final_watermark[final_y_start : final_y_start + (crop_y_end - crop_y_start),
                         final_x_start : final_x_start + (crop_x_end - crop_x_start)] = rotated[
                            crop_y_start:crop_y_end,
                            crop_x_start:crop_x_end
                        ]

        return final_watermark

    def create_hrv_graph_region(self, processed_frame_for_bg, graph_width, graph_height, hr_list, hrv_list, spo2_list, watermark_img=None):
        try:
            video_bg_source = cv2.resize(processed_frame_for_bg, (graph_width, graph_height))
        except cv2.error as e:
            print(f"Error resizing background source frame: {e}")
            video_bg_source = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

        blurred_bg = cv2.GaussianBlur(video_bg_source, self.BLUR_KERNEL_BG, self.BLUR_SIGMA_BG)

        gray_overlay = np.ones_like(blurred_bg) * 220
        blurred_overlay = cv2.GaussianBlur(gray_overlay, self.BLUR_KERNEL_OVERLAY, self.BLUR_SIGMA_OVERLAY)

        graph_region = cv2.addWeighted(blurred_bg, self.BLEND_WEIGHT_BG, blurred_overlay, self.BLEND_WEIGHT_OVERLAY, 0)

        padding = self.GRAPH_PADDING
        inner_graph_height_total = graph_height - padding * 4
        spo2_hrv_height_ratio = 1.2
        pie_chart_height_ratio = 1.2
        total_ratio = spo2_hrv_height_ratio * 2 + pie_chart_height_ratio
        inner_graph_height_spo2_hrv = max(1, int(inner_graph_height_total * spo2_hrv_height_ratio / total_ratio))
        inner_graph_height_pie = max(1, int(inner_graph_height_total * pie_chart_height_ratio / total_ratio))


        inner_graph_width = graph_width - (padding * 2)
        inner_graph_width = max(1, inner_graph_width)

        fig = plt.figure(figsize=(inner_graph_width/100, (inner_graph_height_spo2_hrv * 2 + inner_graph_height_pie)/100), dpi=100)
        fig.patch.set_alpha(0.0)

        text_color = '#2D3436'
        grid_color = '#636E72'
        plot_colors = ['#7CFC00', '#0000FF'] # Green for SpO2, Blue for HRV

        plt.rcParams.update({
            'axes.facecolor': (0.95, 0.95, 0.95, 0.3),
            'axes.edgecolor': grid_color,
            'axes.labelcolor': text_color,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.color': grid_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'font.family': 'sans-serif',
            'font.weight': 'bold'
        })

        gs = fig.add_gridspec(3, 1, height_ratios=[spo2_hrv_height_ratio, spo2_hrv_height_ratio, pie_chart_height_ratio], hspace=0.7)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(spo2_list, color=plot_colors[0], linewidth=2, marker='o', markersize=3)
        ax1.set_title('SpO2 Over Time', fontsize=6, fontweight='bold', color=text_color)
        ax1.set_ylabel('SpO2 (%)', fontsize=4, color=text_color, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.grid(True, alpha=0.2, color=grid_color)
        ax1.set_ylim(min(spo2_list) - 2 if spo2_list else 90, max(spo2_list) + 2 if spo2_list else 100)


        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(hrv_list, color=plot_colors[1], linewidth=2, marker='o', markersize=3)
        ax2.set_title('HRV (RMSSD) Over Time', fontsize=6, fontweight='bold', color=text_color)
        ax2.set_ylabel('HRV (ms)', fontsize=4, color=text_color, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=5)
        ax2.grid(True, alpha=0.2, color=grid_color)
        ax2.set_ylim(min(hrv_list) - 5 if hrv_list else 10, max(hrv_list) + 5 if hrv_list else 80)
        ax2.set_xlabel('Time (sequences)', fontsize=4, color=text_color, fontweight='bold')

        ax3 = fig.add_subplot(gs[2, 0])

        # Calculate HR zone distribution (still based on HR, as per original logic)
        hr_zone_counts = {zone: 0 for zone in HR_ZONES}
        if hr_list:
            for hr in hr_list:
                if hr < 60:
                    hr_zone_counts["Low (<60)"] += 1
                elif 60 <= hr <= 100:
                    hr_zone_counts["Normal (60-100)"] += 1
                else:
                    hr_zone_counts["High (>100)"] += 1

        hr_zone_labels = list(hr_zone_counts.keys())
        hr_zone_sizes = list(hr_zone_counts.values())
        hr_zone_colors = [HR_ZONES.get(label, 'grey') for label in hr_zone_labels]

        if hr_zone_sizes and sum(hr_zone_sizes) > 0:
            # Increased radius and added labels to the pie chart
            wedges, texts, autotexts = ax3.pie(hr_zone_sizes, labels=hr_zone_labels, colors=hr_zone_colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 6, 'color': text_color, 'fontweight': 'bold'}, radius=1.8)
            ax3.axis('equal')
            ax3.set_title('Heart Rate Zone Distribution', fontsize=6, fontweight='bold', color=text_color)
            # Removed legend as labels are inside
        else:
            ax3.text(0.5, 0.5, "No HR data available", horizontalalignment='center', verticalalignment='center', fontsize=6, color=text_color, fontweight='bold', transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)


        plt.tight_layout(rect=[0, 0, 1, 1])

        canvas = FigureCanvas(fig)
        canvas.draw()

        plot_img_rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        plot_img_rgba = plot_img_rgba.reshape(canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        plot_img_resized = cv2.resize(plot_img_rgba, (inner_graph_width, inner_graph_height_spo2_hrv * 2 + inner_graph_height_pie))

        y_start_plots = padding
        x_start_plots = padding

        roi_plots = graph_region[y_start_plots : y_start_plots + inner_graph_height_spo2_hrv * 2 + inner_graph_height_pie,
                                 x_start_plots : x_start_plots + inner_graph_width]

        if roi_plots.shape[:2] == plot_img_resized.shape[:2]:
            alpha_plot = plot_img_resized[:, :, 3] / 255.0
            alpha_plot = np.dstack([alpha_plot] * 3)
            graph_region[y_start_plots : y_start_plots + inner_graph_height_spo2_hrv * 2 + inner_graph_height_pie,
                         x_start_plots : x_start_plots + inner_graph_width] = (
                                 roi_plots * (1 - alpha_plot) +
                                 plot_img_resized[:, :, :3] * alpha_plot
                               ).astype(np.uint8)


        if watermark_img is not None:
            watermark_sized = cv2.resize(watermark_img, (graph_width, graph_height))
            if watermark_sized.shape[2] == 4:
                alpha_w = watermark_sized[:, :, 3] / 255.0
                alpha_w = np.dstack([alpha_w] * 3)
                graph_region = (graph_region * (1 - alpha_w * 0.4) +
                                 watermark_sized[:, :, :3] * (alpha_w * 0.4)).astype(np.uint8)


        return graph_region


    def create_bordered_frame(self, main_content, heading_texts, output_width, output_height, heading_height, frame_width, processed_panel_width, graph_panel_width, metrics=None):
        final_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        actual_titlebar_height = 40
        content_y_start = heading_height
        titlebar = np.ones((actual_titlebar_height, output_width, 3), dtype=np.uint8) * 255

        gradient = np.linspace(0.97, 1.0, actual_titlebar_height)[:, np.newaxis, np.newaxis]
        titlebar = (titlebar * gradient).astype(np.uint8)

        cv2.line(titlebar, (0, actual_titlebar_height-1), (output_width, actual_titlebar_height-1),
                 (240, 240, 240), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        text_color = (70, 70, 70)
        shadow_color = (200, 200, 200)
        text_thickness = 2

        temp_text_size = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_y = (actual_titlebar_height + temp_text_size[1]) // 2

        text_size_orig = cv2.getTextSize(heading_texts[0], font, font_scale, text_thickness)[0]
        text_x_orig = (frame_width - text_size_orig[0]) // 2

        cv2.putText(titlebar, heading_texts[0],
                     (text_x_orig+1, text_y+1),
                     font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[0],
                     (text_x_orig, text_y),
                     font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        right_panels_start_x = frame_width
        right_panels_width = processed_panel_width + graph_panel_width
        text_size_foai = cv2.getTextSize(heading_texts[1], font, font_scale, text_thickness)[0]
        text_x_foai = right_panels_start_x + (right_panels_width - text_size_foai[0]) // 2

        cv2.putText(titlebar, heading_texts[1],
                     (text_x_foai+1, text_y+1),
                     font, font_scale, shadow_color, text_thickness, cv2.LINE_AA)
        cv2.putText(titlebar, heading_texts[1],
                     (text_x_foai, text_y),
                     font, font_scale, text_color, text_thickness, cv2.LINE_AA)


        if metrics:
            current_emotion, hrv, sdnn, spo2 = metrics
            font_size_metrics = 0.3
            text_thickness_metrics = 1
            shadow_color = (200, 200, 200)
            text_color_blue = (255, 0, 0)

            metric_area_start_x = frame_width + processed_panel_width
            metric_area_width = graph_panel_width
            metric_padding_x = int(metric_area_width * 0.05)
            metric_padding_y = 5
            col_gap = int(metric_area_width * 0.10)  # Gap between columns

            # Column 1: HRV and SDNN
            col1_x = metric_area_start_x + metric_padding_x
            col1_y1 = metric_padding_y + 10
            col1_y2 = col1_y1 + 15

            hrv_text = f"HRV: {hrv:.1f}ms"
            sdnn_text = f"SDNN: {sdnn:.1f}ms"

            cv2.putText(titlebar, hrv_text,
                         (col1_x + 1, col1_y1 + 1),
                         font, font_size_metrics, shadow_color, text_thickness_metrics, cv2.LINE_AA)
            cv2.putText(titlebar, hrv_text,
                         (col1_x, col1_y1),
                         font, font_size_metrics, text_color_blue, text_thickness_metrics, cv2.LINE_AA)

            cv2.putText(titlebar, sdnn_text,
                         (col1_x + 1, col1_y2 + 1),
                         font, font_size_metrics, shadow_color, text_thickness_metrics, cv2.LINE_AA)
            cv2.putText(titlebar, sdnn_text,
                         (col1_x, col1_y2),
                         font, font_size_metrics, text_color_blue, text_thickness_metrics, cv2.LINE_AA)

            # Column 2: Emotion and SpO2
            col2_x = col1_x + col_gap + 60  # Adjust 90 for text width if needed
            col2_y1 = col1_y1
            col2_y2 = col1_y2

            emotion_text = f"Emotion: {current_emotion}"
            emotion_font_size = 0.2
            spo2_text = f"SpO2: {spo2:.1f}%"

            cv2.putText(titlebar, emotion_text,
                         (col2_x + 1, col2_y1 + 1),
                         font, emotion_font_size, shadow_color, text_thickness_metrics, cv2.LINE_AA)
            cv2.putText(titlebar, emotion_text,
                         (col2_x, col2_y1),
                         font, emotion_font_size, text_color_blue, text_thickness_metrics, cv2.LINE_AA)

            cv2.putText(titlebar, spo2_text,
                         (col2_x + 1, col2_y2 + 1),
                         font, font_size_metrics, shadow_color, text_thickness_metrics, cv2.LINE_AA)
            cv2.putText(titlebar, spo2_text,
                         (col2_x, col2_y2),
                         font, font_size_metrics, text_color_blue, text_thickness_metrics, cv2.LINE_AA)


        final_frame[0:actual_titlebar_height, :] = titlebar

        content_y = actual_titlebar_height + 5
        content_height, content_width = main_content.shape[:2]
        if content_y + content_height <= output_height and content_width <= output_width:
            final_frame[content_y : content_y + content_height, :content_width] = main_content
        else:
            pass


        return final_frame

    def estimate_emotion(self, hr, hrv):
        hr = float(hr)
        hrv = float(hrv)

        if hr > 120 and hrv < 20:
                 return "Fearful"
        elif hr > 100 and hrv > 50:
            return "Excited"
        elif hr < 70 and hrv > 50:
            return "Happy"
        elif hr > 100 and hrv < 30:
            return "Stressed"
        elif hr > 90 and hrv < 30:
            return "Anxious"
        elif hr < 70 and hrv < 30:
            return "Sad"
        else:
            return "Neutral"

    # Emotion classification based on SpO2
    def classify_emotion(self, spo2_value):
        if spo2_value is None:
             return "Analyzing..."
        if spo2_value > 97:
            return "Calm / Relaxed"
        elif 95 < spo2_value <= 97:
            return "Neutral / Normal"
        elif 92 < spo2_value <= 95:
            return "Moderate Excitement"
        elif 90 < spo2_value <= 92:
            return "High Excitement / Stress"
        elif spo2_value <= 90:
            return "Extreme Arousal / Panic"
        else:
            return "Uncertain / Mixed State"


    def predict_ibi_from_video(self, video_path, output_path):
        # Initialize report data
        report_data = [['Frame No.', 'Timestamp', 'Heart Rate (BPM)', 'HRV (ms)', 'SpO2 (%)', 'Emotional State']]
        
        if self.model is None or self.face_cascade is None:
            return 0, 0, "Error: Model or face cascade not loaded.", 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0, "Error: Cannot open video file", 0

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        processed_panel_width = int(fw * 0.7)
        graph_panel_width = int(fw * 0.3)
        if fw < 100:
             processed_panel_width = max(50, processed_panel_width)
             graph_panel_width = max(50, graph_panel_width)


        output_width = fw + processed_panel_width + graph_panel_width
        output_width += output_width % 2

        heading_height = 46
        output_height = fh + heading_height
        output_height += output_height % 2

        graph_width = graph_panel_width
        graph_height = fh

        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_output_video = os.path.join(temp_dir, f"temp_hrv_annotated_{timestamp}.mp4")

        watermark_path = os.path.join(settings.STATIC_ROOT, "testapp", "images", "watermark.png")
        watermark = self.prepare_watermark_hrv(watermark_path, graph_width, graph_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_video, fourcc, fps, (output_width, output_height))

        roi_buffer = []
        frame_buffer = []
        spo2_predictions = []
        emotion_predictions = []
        hr_list, hrv_list, all_ibi_values = [], [], []
        current_hr_to_display = 0.0
        current_hrv_to_display = 0.0
        current_emotion_to_display = "Analyzing..."
        current_spo2_to_display = None # Initialize as None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            annotated_frame = frame.copy()

            face_roi, face_coords = self.extract_face_roi(annotated_frame)

            # Process for SpO2 prediction
            spo2_roi = self.extract_face_roi_spo2(original_frame, self.roi_size)
            if spo2_roi is not None:
                frame_buffer.append(spo2_roi)
            if len(frame_buffer) > self.squence_length:
                frame_buffer.pop(0)  # Keep only the last N frames

            if len(frame_buffer) == self.squence_length and self.spo2_model is not None:
                try:
                    seq = np.array(frame_buffer).astype(np.float32) / 255.0
                    seq = np.expand_dims(seq, axis=0)
                    pred_spo2 = self.spo2_model.predict(seq, verbose=0)[0][0]
                    spo2_predictions.append(pred_spo2)

                    current_spo2_to_display = pred_spo2

                    emotion = self.classify_emotion(pred_spo2)
                    emotion_predictions.append(emotion)

                    current_emotion_to_display = emotion
                except Exception as e:
                     print(f"Error predicting SpO2 or emotion: {e}")
                     current_spo2_to_display = None # Reset on error
                     current_emotion_to_display = "Error"
                     pass


            # Process for IBI prediction (HRV/HR)
            if face_coords is not None:
                x, y, w, h = face_coords
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if face_roi is not None:
                    roi_buffer.append(face_roi / 255.0)

            if len(roi_buffer) == self.FRAMES_PER_SEQUENCE and self.model is not None:
                try:
                    input_sequence = np.expand_dims(np.array(roi_buffer), axis=0)
                    predicted_ibi_sequence = self.model.predict(input_sequence, verbose=0)[0] * 1000

                    hr_seq = self.compute_hr(predicted_ibi_sequence)
                    hrv_seq = self.compute_rmssd(predicted_ibi_sequence)

                    # Add data to report
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(frame_idx/fps))
                    report_data.append([
                        str(frame_idx),
                        timestamp,
                        f"{current_hr_to_display:.1f}",
                        f"{current_hrv_to_display:.1f}",
                        f"{current_spo2_to_display:.1f}" if current_spo2_to_display else "N/A",
                        current_emotion_to_display
                    ])

                    roi_buffer = [] # Clear buffer after prediction

                    hr_list.append(hr_seq["avg_heart_rate"])  # ✅ Only the float is needed
                    hrv_list.append(hrv_seq)
                    all_ibi_values.extend(predicted_ibi_sequence)


                    current_hr_to_display = hr_seq["avg_heart_rate"]  # ✅ Extract the value
                    current_hrv_to_display = hrv_seq

                except Exception as e:
                    print(f"Error predicting IBI/HR/HRV: {e}")
                    current_hr_to_display = 0.0
                    current_hrv_to_display = 0.0
                    pass


            new_w_proc, new_h_proc, x_offset_proc, y_offset_proc = self.calculate_scaled_dimensions(
                 fw, fh, processed_panel_width, fh
            )
            processed_panel_frame = np.zeros((fh, processed_panel_width, 3), dtype=np.uint8)
            try:
                annotated_frame_resized = cv2.resize(annotated_frame, (new_w_proc, new_h_proc))
                processed_panel_frame[y_offset_proc : y_offset_proc + new_h_proc,
                                      x_offset_proc : x_offset_proc + new_w_proc] = annotated_frame_resized
            except cv2.error as e:
                print(f"Error resizing annotated frame: {e}")


            graph_region = self.create_hrv_graph_region(
                 processed_panel_frame,
                 graph_width,
                 graph_height,
                 hr_list, # Still needed for the pie chart
                 hrv_list,
                 spo2_predictions, # Pass spo2_predictions here
                 watermark
            )

            if original_frame.shape[0] != fh or original_frame.shape[1] != fw:
                 original_frame = cv2.resize(original_frame, (fw, fh))
            if processed_panel_frame.shape[0] != fh or processed_panel_frame.shape[1] != processed_panel_width:
                 processed_panel_frame = cv2.resize(processed_panel_frame, (processed_panel_width, fh))
            if graph_region.shape[0] != fh or graph_region.shape[1] != graph_panel_width:
                 graph_region = cv2.resize(graph_region, (graph_panel_width, fh))


            main_content = cv2.hconcat([original_frame, processed_panel_frame, graph_region])

            final_hrv_to_display = hrv_list[-1] if hrv_list else 0.0
            overall_sdnn_to_display = self.compute_sdnn(all_ibi_values)
            # Use the latest individual spo2 prediction for display in the titlebar
            current_spo2_for_display = spo2_predictions[-1] if spo2_predictions else (current_spo2_to_display if current_spo2_to_display is not None else 0.0)


            final_frame = self.create_bordered_frame(
                 main_content,
                 ["Original Video", "FOAI"],
                 output_width,
                 output_height,
                 heading_height,
                 fw,
                 processed_panel_width,
                 graph_panel_width,
                 metrics=(current_emotion_to_display, final_hrv_to_display, overall_sdnn_to_display, current_spo2_for_display)
            )


            out.write(final_frame)

            frame_idx += 1

        cap.release()
        out.release()

        # Compute final metrics
        hr_metrics = self.compute_hr(all_ibi_values)
        avg_resp_rate = self.compute_avg_respiration_rate(all_ibi_values)
        final_sdnn = self.compute_sdnn(all_ibi_values)
        average_spo2 = np.mean(spo2_predictions) if spo2_predictions else 0.0
        final_mean_hr = hr_metrics["avg_heart_rate"]
        self.avg_heart_rate = final_mean_hr
        self.max_heart_rate = hr_metrics["max_heart_rate"]
        self.min_heart_rate = hr_metrics["min_heart_rate"]
        self.heart_rates = hr_metrics["heart_rates"]
        self.avg_respiration_rate = avg_resp_rate

        most_common_emotion = "N/A"
        if emotion_predictions:
            emotion_counter = Counter(emotion_predictions)
            most_common_emotion, count = emotion_counter.most_common(1)[0]
            print(f"Most common emotion: {most_common_emotion} with count: {count}")


        temp_output_for_encode = temp_output_video
        final_encoded_output_path = output_path

        print(f"Encoding video from {temp_output_for_encode} to {final_encoded_output_path} with audio from {video_path}")
        encoded_path_result = self.encode_video(temp_output_for_encode, final_encoded_output_path, video_path)

        try:
             if os.path.exists(temp_output_for_encode):
                 os.remove(temp_output_for_encode)
                 print(f"Removed temporary file: {temp_output_for_encode}")
        except Exception as e:
             print(f"Error removing temporary OpenCV video file: {e}")


        if encoded_path_result is None:
            return 0, 0, "Error: Failed to encode final video.", 0

        # Generate PDF report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports_dir = os.path.join(settings.MEDIA_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_filename = f"hrv_spO2_report_{timestamp}.pdf"
        report_path = os.path.join(reports_dir, report_filename)
        
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        elements = []
        
        # Add title and summary
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Heart Rate Variability and SpO2 Analysis Report", styles['Heading1']))
        elements.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Total Frames Analyzed: {len(report_data)-1}", styles['Normal']))
        
        # After adding other report content
        elements.append(Paragraph("Heart Rate Variability and SpO2 Analysis Report", styles['Heading1']))
        elements.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Total Frames Analyzed: {len(report_data)-1}", styles['Normal']))
        elements.append(Paragraph(f"Mean Heart Rate: {final_mean_hr:.1f} BPM", styles['Normal']))
        elements.append(Paragraph(f"Max Heart Rate: {self.max_heart_rate:.1f} BPM", styles['Normal']))
        elements.append(Paragraph(f"Min Heart Rate: {self.min_heart_rate:.1f} BPM", styles['Normal']))
        elements.append(Paragraph(f"SDNN: {final_sdnn:.1f} ms", styles['Normal']))
        elements.append(Paragraph(f"Average Respiration Rate: {self.avg_respiration_rate:.1f} breaths/min", styles['Normal']))
        elements.append(Paragraph(f"Average SpO2: {average_spo2:.1f}%", styles['Normal']))
        elements.append(Paragraph(f"Dominant Emotion: {most_common_emotion}", styles['Normal']))

        # Before generating PDF report, get the emotion reason
        emotion_reason = self.generate_emotion_reason(final_mean_hr, final_sdnn, most_common_emotion)
        
        # Add emotion reasoning section
        elements.append(Paragraph("<br/>Emotional Analysis:", styles['Heading2']))
        elements.append(Paragraph(emotion_reason, styles['Normal']))
        
        # Add emotion distribution
        elements.append(Paragraph("<br/>Emotion Distribution:", styles['Heading2']))
        emotion_counts = Counter(emotion_predictions)
        total_frames = sum(emotion_counts.values())
        for emotion, count in emotion_counts.items():
            if total_frames > 0:
                percentage = (count / total_frames) * 100
                elements.append(Paragraph(f"{emotion}: {percentage:.1f}%", styles['Normal']))
        
        elements.append(Paragraph("<br/><br/>", styles['Normal']))
        
        # Create table
        table = Table(report_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        
        # Generate PDF
        doc.build(elements)
    
        return final_sdnn, final_mean_hr, most_common_emotion, average_spo2, report_filename

    def encode_video(self, input_path_video, output_path, input_path_audio=None):
        try:
            command = [
                'ffmpeg', '-i', input_path_video,
            ]
            if input_path_audio and os.path.exists(input_path_audio):
                 command.extend(['-i', input_path_audio])

            command.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
            ])

            if input_path_audio and os.path.exists(input_path_audio):
                command.extend([
                    '-c:a', 'aac',
                    '-strict', '-2',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                ])
            else:
                 # Add a silent audio track if the input video has no audio
                 command.extend([
                      '-f', 'lavfi',
                      '-i', 'anullsrc=cl=stereo:r=44100',
                      '-c:a', 'aac',
                      '-map', '0:v:0',
                      '-map', '1:a:0',
                      '-shortest' # ensure output duration matches video if audio is longer
                 ])


            command.extend([
                '-movflags', '+faststart',
                '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',
                '-y',
                output_path
            ])

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding error:\n{process.stderr.decode()}")
                return None

            print(f"FFmpeg encoding successful: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error during FFmpeg encoding: {e}")
            return None
    
    def post(self, request):
        if self.model is None or self.face_cascade is None or self.spo2_model is None:
            error_message = "Analysis models not loaded. "
            if self.model is None: error_message += "HRV model missing. "
            if self.face_cascade is None: error_message += "Face cascade missing. "
            if self.spo2_model is None: error_message += "SpO2 model missing. "
            return JsonResponse({"error": error_message.strip()}, status=500)


        try:
            video_path = request.session.get("uploaded_video_path")
            if not video_path:
                return JsonResponse({"error": "No video file found in session"}, status=400)

            video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
            if not os.path.exists(video_absolute_path):
                return JsonResponse({"error": "Video file not found on server"}, status=404)

            output_dir = os.path.join(settings.MEDIA_ROOT, "output_videos")
            os.makedirs(output_dir, exist_ok=True)
            video_uuid = request.session.get("video_uuid", None)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_output_name = f"hrv_output_{video_uuid}_{timestamp}.mp4"

            output_path = os.path.join(output_dir, final_output_name)

            sdnn, mean_hr, overall_emotion, average_spo2, report_filename = self.predict_ibi_from_video(video_absolute_path, output_path)

            if "Error" in overall_emotion or "Error" in str(sdnn):
                 return JsonResponse({"error": f"Analysis failed: {overall_emotion}" if "Error" in overall_emotion else "Analysis failed.",
                                       "details": f"SDNN: {sdnn}, Mean HR: {mean_hr}, Avg SpO2: {average_spo2}"}, status=500)

            content = {
                'video_path': f"{settings.MEDIA_URL}output_videos/{final_output_name}",
                'sdnn': float(sdnn),
                'mean_hr': float(mean_hr),
                'status': 'success',
                'avg_emotion': overall_emotion,
                'average_spo2': float(average_spo2),
                'report_path': f"{settings.MEDIA_URL}reports/{report_filename}"
            }

            return JsonResponse(content)

        except FileNotFoundError as e:
            print(f"File error in HRV analysis: {str(e)}")
            return JsonResponse({
                "error": "A required file was not found.",
                "details": str(e),
                "status": "error"
            }, status=404)
        except Exception as e:
            print(f"Error in HRV analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                "error": "An unexpected error occurred during analysis.",
                "details": str(e),
                "status": "error"
            }, status=500)

    def generate_emotion_reason(self, mean_hr, sdnn, avg_emotion):
        reason = f"Based on a mean HR of {mean_hr:.1f} bpm and SDNN of {sdnn:.1f} ms, "
        if avg_emotion == "Excited":
            return reason + "a high heart rate and high variability indicate strong positive arousal."
        elif avg_emotion == "Happy":
            return reason + "a low heart rate and high variability suggest a calm and positive emotional state."
        elif avg_emotion == "Stressed":
            return reason + "a high heart rate with low variability is often linked to stress."
        elif avg_emotion == "Anxious":
            return reason + "elevated heart rate and reduced variability commonly reflect anxiety."
        elif avg_emotion == "Neutral":
            return reason + "heart rate and variability lie within typical resting range, implying a neutral state."
        elif avg_emotion == "Sad":
            return reason + "a low heart rate and low variability suggest a subdued or sad emotional condition."
        elif avg_emotion == "Fearful":
            return reason + "a very high heart rate and extremely low variability are associated with fear."
        else:
            return reason + "the metrics suggest a balanced physiological state."


###################### HeartRate class end ########################



###################### Speech class #######################
'''# class Speech(View):
#     def __init__(self):
#         super().__init__()

#     def detect_language(self,audio_path):
#         model = whisper.load_model("base")
#         result = model.transcribe(audio_path, task="transcribe")
#         lang_code = result.get("language", "en")
#         return lang_code, result["text"]
    
#     def preprocess_text(self,text):
#         sentences = re.split(r'[.!?]', text)
#         return [s.strip() for s in sentences if s.strip()]
    
#     def analyze_sentiment(self,sentences, lang_code):
#         if lang_code == "hi":
#             model_name = "LondonStory/txlm-roberta-hindi-sentiment"
#             sentiment_labels = ['Negative', 'Neutral', 'Positive']
#         else:
#             model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#             sentiment_labels = ['Negative', 'Neutral', 'Positive']

#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)

#         sentence_results = []
#         sentiment_counts = [0, 0, 0]
#         confidences = []

#         for sentence in sentences:
#             inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
#             with torch.no_grad():
#                 logits = model(**inputs).logits
#             scores = softmax(logits.cpu().numpy()[0])
#             label = np.argmax(scores)
#             confidence = float(scores[label])

#             sentence_results.append({
#                 'sentence': sentence,
#                 'label': sentiment_labels[label],
#                 'confidence': confidence
#             })

#             sentiment_counts[label] += 1
#             confidences.append(confidence)

#         total = sum(sentiment_counts)
#         majority_label = sentiment_labels[np.argmax(sentiment_counts)]
#         average_conf = np.mean(confidences)

#         return majority_label, average_conf, sentiment_counts, confidences, sentence_results

#     def plot_sentiment_bar(self,sentiment_counts):
#         labels = ['Negative', 'Neutral', 'Positive']
#         plt.figure(figsize=(6, 4))
#         plt.bar(labels, sentiment_counts, color=['red', 'gray', 'green'])
#         plt.title("Sentiment Distribution")
#         plt.ylabel("Number of Sentences")
#         plt.show()

#     def plot_confidence_distribution(self,confidences):
#         plt.figure(figsize=(6, 4))
#         plt.hist(confidences, bins=10, color='blue', alpha=0.7)
#         plt.title("Confidence Score Distribution")
#         plt.xlabel("Confidence")
#         plt.ylabel("Frequency")
#         plt.show()

#     def convert_audio_to_wav(self, audio_path, wav_path="converted_audio.wav"):
#         print(f"Converting audio {audio_path} to 16kHz mono WAV...")
#         command = [
#             "ffmpeg",
#             "-y",  # overwrite output file if exists
#             "-i", audio_path,
#             "-ac", "1",
#             "-ar", "16000",
#             wav_path
#         ]
#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         return wav_path

#     def post(self, request):
#         audio_path = request.session.get('extracted_audio_path')
#         if not audio_path:
#             return JsonResponse({"error": "Audio file not found in session"}, status=400)

#         audio_absolute_path = os.path.join(settings.MEDIA_ROOT, audio_path)
#         if not os.path.exists(audio_absolute_path):
#             return JsonResponse({"error": "Audio file not found on server"}, status=404)
        
#         # convert audio to WAV if needed
#         if not audio_absolute_path.endswith('.wav'):
#             audio_path = self.convert_audio_to_wav(audio_absolute_path)
#         else:  
#             audio_path = audio_absolute_path

#         print("🔍 Detecting language and transcribing...")
#         lang_code, transcript = self.detect_language(audio_path)

#         print(f"\n🌐 Detected Language: {lang_code.upper()}")
#         print("\n📄 Transcript:\n", transcript)

#         sentences = self.preprocess_text(transcript)
#         print(f"\n🧠 {len(sentences)} Sentences Detected.")

#         final_sentiment, avg_conf, sentiment_counts, confidences, sentence_results = self.analyze_sentiment(sentences, lang_code)

#         print("\n✅ Final Sentiment:", final_sentiment)
#         print("📈 Average Confidence:", round(avg_conf, 3))

#         print("\n🧪 Per Sentence Analysis:")
#         for res in sentence_results:
#             print(f"- {res['label']:>8} ({res['confidence']:.2f}): {res['sentence']}")


#         # plot_sentiment_bar(sentiment_counts)
#         # plot_confidence_distribution(confidences)

#             return JsonResponse({
#                 "sentiment": final_sentiment,
#                 "confidence": confidences,
#                 "language": lang_code,
#             })
'''
# ---------------- speech v2 calss -------------------

class Speech_V2(View):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_device = torch.device("cpu")
        self.whisper_model = whisper.load_model("large-v3", device=self.whisper_device)

    def convert_audio_to_wav(self, input_path, output_path="converted_audio.wav"):
        print(f"🎧 Converting {input_path} to 16kHz mono WAV with noise reduction...")
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", input_path,
            "-ac", "1",         # Mono
            "-ar", "16000",     # 16kHz sample rate
            "-af", "afftdn, highpass=f=200, lowpass=f=3000",  # Noise reduction
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def detect_language_and_transcribe(self, wav_path, temperature=0, beam_size=5):
        print("🔍 Loading and preprocessing audio...")
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.whisper_device)

        _, lang_probs = self.whisper_model.detect_language(mel)
        detected_lang = max(lang_probs, key=lang_probs.get)
        lang_code = "hi" if detected_lang == "hi" else "en"

        result = self.whisper_model.transcribe(
            wav_path,
            language=lang_code,
            temperature=temperature,
            beam_size=beam_size
        )

        return lang_code, result["text"]

    def preprocess_text(self, text):
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def analyze_sentiment(self, sentences, lang_code):
        if lang_code == "hi":
            model_name = "LondonStory/txlm-roberta-hindi-sentiment"
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
        else:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment"
            sentiment_labels = ['Negative', 'Neutral', 'Positive']

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        sentence_results = []
        sentiment_counts = [0, 0, 0]
        confidences = []

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            scores = softmax(logits.cpu().numpy()[0])
            label = np.argmax(scores)
            confidence = float(scores[label])

            sentence_results.append({
                'sentence': sentence,
                'label': sentiment_labels[label],
                'confidence': confidence
            })

            sentiment_counts[label] += 1
            confidences.append(confidence)

        majority_label = sentiment_labels[np.argmax(sentiment_counts)]
        average_conf = np.mean(confidences)

        return majority_label, average_conf, sentiment_counts, confidences, sentence_results

    def analysis(self, audio_input_path, audio_output_path="converted_audio.wav"):
        print(" Detecting language and transcribing...")
        wav_path = self.convert_audio_to_wav(audio_input_path, audio_output_path)
        lang_code, transcript = self.detect_language_and_transcribe(wav_path)

        print(f"\n Detected Language: {lang_code.upper()}")
        print("\n Transcript:\n", transcript)

        sentences = self.preprocess_text(transcript)
        print(f"\n {len(sentences)} Sentences Detected.")

        final_sentiment, avg_conf, sentiment_counts, confidences, sentence_results = self.analyze_sentiment(sentences, lang_code)

        print("\n Final Sentiment:", final_sentiment)
        print(" Average Confidence:", round(avg_conf, 3))

        print("\n Per Sentence Analysis:")
        for res in sentence_results:
            print(f"- {res['label']:>8} ({res['confidence']:.2f}): {res['sentence']}")

        return lang_code,final_sentiment,avg_conf

    def post(self,request):
        audio_path = request.session.get('extracted_audio_path')
        if not audio_path:
            return JsonResponse({"error": "Audio file not found in session"}, status=400)

        audio_absolute_path = os.path.join(settings.MEDIA_ROOT, audio_path)
        if not os.path.exists(audio_absolute_path):
            return JsonResponse({"error": "Audio file not found on server"}, status=404)
        
        # convert audio to WAV if needed
        if not audio_absolute_path.endswith('.wav'):
            audio_path = self.convert_audio_to_wav(audio_absolute_path)
        else:  
            audio_path = audio_absolute_path

        language , sentiment , confidence = self.analysis(audio_path)

        return JsonResponse({
            "sentiment": sentiment,
            "confidence": confidence,
            "language": language,
        })


# Scikit-learn for metrics (optional, with manual fallbacks)
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# --- GLOBAL SETUP & CONFIGURATIONS (As before, ideally in settings.py) ---
IS_COLAB = False
# Logging setup - integrate with Django's logging or keep simple for this file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if SKLEARN_AVAILABLE:
    logging.info("scikit-learn found. Will use for advanced metrics.")
else:
    logging.warning("scikit-learn not found. Metrics will be basic (accuracy). Install sklearn for detailed reports.")

SCRIPT_VERSION = "V2.5"
SHAPE_PREDICTOR_FILENAME: str = "shape_predictor_68_face_landmarks.dat"
PDF_REPORT_FILENAME_SUFFIX: str = f"_eye_analysis_report_{SCRIPT_VERSION.lower().replace(' ', '_')}.pdf"

DEFAULT_EAR_THRESHOLD_BLINK: float = 0.23
DEFAULT_EAR_CONSEC_FRAMES_BLINK: int = 2
DEFAULT_EAR_THRESHOLD_DROWSY: float = 0.15
DEFAULT_FIXATION_MAX_MOVEMENT_PX: int = 8
DEFAULT_T_BLINK_RATE_NERVOUS_HIGH_STATE: float = 30.0
DEFAULT_FIXATION_STABILITY_MAX_VARIANCE: float = 50.0
DEFAULT_T_FIX_STABILITY_GOOD_STATE_FACTOR: float = 0.7
DEFAULT_T_SACCADE_AMP_SCANNING_HIGH_STATE: float = 25.0

BLINK_RATE_WINDOW_SECONDS: int = 20
FIXATION_STATS_WINDOW_SECONDS: int = 15
EAR_CONSEC_FRAMES_DROWSY: int = 45
MIN_FRAMES_FOR_FIXATION: int = 6
FIXATION_STABILITY_WINDOW_FRAMES: int = 10
IPD_CHANGE_REL_THRESHOLD_SIGNIFICANT: float = 0.05
T_BLINK_RATE_FOCUSED_LOW_STATE: float = 12.0
BASE_DURATION_FIX_FOCUSED_SHORT_S: float = 0.5
BASE_DURATION_FIX_FOCUSED_INTENSE_S: float = 1.8
T_FIX_STABILITY_POOR_STATE_FACTOR: float = 1.5
T_SACCADE_AMP_JITTER_LOW_STATE: float = 5.0
BASE_T_IPD_D1_DILATE_STRONG_FACTOR: float = 0.03
BASE_T_IPD_D1_CONSTRICT_STRONG_FACTOR: float = -0.03
T_IODR_HIGH_ENGAGE_STATE: float = 0.165
T_IODR_LOW_SQUINT_STATE: float = 0.125
T_HEAD_ROLL_THINKING_STATE: float = 12.0
T_HEAD_YAW_THINKING_MAX_STATE: float = 15.0
T_SACCADE_AMP_THINKING_MAX_STATE: float = 7.0
DEFAULT_PROCESSING_FPS: float = 30.0
MAX_PROCESSING_FPS: float = 30.0
FRAME_SKIP: int = 1

RULE_BASED_SMOOTHING_WINDOW_SIZE: int = 5

DEJAVU_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEJAVU_FONT_ITALIC_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"

EMOTION_CLASSES_V2_5 = [
    "Engaged/Focused", "Distracted/Scanning", "Tired/Fatigued",
    "Agitated/Stressed", "Neutral/Calm", "Blinking", "Pondering", "No Face/Unknown"
]
NUM_ML_EMOTION_CLASSES_V2_5 = len(EMOTION_CLASSES_V2_5)
STATE_TO_EMOTION_MAP_V2_5 = {
    "Blinking": "Blinking", "Fatigued_Drowsy": "Tired/Fatigued",
    "Fatigued_Subtle": "Tired/Fatigued", "Focused_Intense": "Engaged/Focused",
    "Focused_Deep": "Engaged/Focused", "Focused_Calm": "Neutral/Calm",
    "Focused_Attentive": "Engaged/Focused", "Nervous_Anxious": "Agitated/Stressed",
    "Nervous_Jittery": "Agitated/Stressed", "Distracted_Scanning": "Distracted/Scanning",
    "Thinking_Pondering": "Pondering", "Natural_Neutral": "Neutral/Calm",
    "No Face": "No Face/Unknown"
}
DEFAULT_EMOTION_V2_5 = "Neutral/Calm"

ML_INPUT_DIM = 20
ML_SEQ_LENGTH = 30
ML_HIDDEN_DIM_LSTM = 64
ML_NUM_LAYERS_LSTM = 1
ML_DROPOUT_RATE_MODEL = 0.2
ML_BATCH_SIZE = 8
ML_LEARNING_RATE = 0.005
ML_NUM_EPOCHS_POC = 5
ML_OPTIMIZER_WEIGHT_DECAY = 1e-5
ML_DUMMY_NUM_TRAIN_SAMPLES = 100
ML_DUMMY_NUM_VAL_SAMPLES = 30
ML_DUMMY_NUM_TEST_SAMPLES = 30

ML_AUGMENT_NOISE_LEVEL = 0.01
ML_AUGMENT_SCALE_RANGE = (0.95, 1.05)

HYBRID_CONFIDENCE_THRESHOLD_ML = 0.7
HYBRID_CONFIDENCE_THRESHOLD_RULE = 0.6

L_START_IDX, L_END_IDX = 36, 42
R_START_IDX, R_END_IDX = 42, 48
NOSE_TIP_IDX, CHIN_IDX = 30, 8
L_EYE_CORNER_IDX, R_EYE_CORNER_IDX = 36, 45
L_MOUTH_CORNER_IDX, R_MOUTH_CORNER_IDX = 48, 54
HEAD_POSE_MODEL_POINTS_3D: np.ndarray = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

# --- Placeholder NIA Implementations (Simulated offline tuning) ---
class PlaceholderNIA:
    def __init__(self, pop_size=10, n_iterations=20, problem_dict: Optional[Dict] = None, verbose=False, **kwargs):
        self.pop_size = max(1, pop_size)
        self.n_iterations = max(1, n_iterations)
        self.problem_dict = problem_dict if problem_dict else {}
        self.verbose = verbose
        self.solution: Optional[List[float]] = None
        self.fitness: float = float('inf')
        self.name = self.__class__.__name__
        logging.debug(f"{self.name} Initialized: PopSize={self.pop_size}, Iterations={self.n_iterations}")

    def solve(self, problem_dict_override: Optional[Dict] = None):
        current_problem = problem_dict_override if problem_dict_override else self.problem_dict
        if not current_problem or not all(k in current_problem for k in ['fit_func', 'lb', 'ub']):
            logging.error(f"{self.name}: Problem dictionary is incomplete or missing. Required keys: 'fit_func', 'lb', 'ub'.")
            return None, float('inf')

        lb = current_problem['lb']
        if not isinstance(lb, list) or not lb:
            logging.error(f"{self.name}: Lower bounds 'lb' must be a non-empty list.")
            return None, float('inf')
        dim = len(lb)
        if dim == 0:
            logging.error(f"{self.name}: Dimension of the problem (length of 'lb') is 0. Cannot optimize.")
            return None, float('inf')

        return self.optimize(current_problem['fit_func'], lb, current_problem['ub'], dim)

    def optimize(self, fitness_func: Callable, lb: List[float], ub: List[float], dim: int):
        logging.info(f"{self.name}: Starting optimization for {dim} dimensions over {self.n_iterations} iterations with population {self.pop_size}.")
        best_sol_candidate: Optional[List[float]] = None
        best_fit_candidate: float = float('inf')

        for i in range(self.n_iterations):
            iter_solutions = []
            iter_fitnesses = []
            for pop_idx in range(self.pop_size):
                try:
                    if len(lb) != dim or len(ub) != dim:
                        logging.error(f"{self.name}: Dimension mismatch between lb/ub and dim during candidate generation.")
                        continue
                    candidate_solution = [random.uniform(lb[j], ub[j]) for j in range(dim)]
                    fit_val = fitness_func(candidate_solution)
                    iter_solutions.append(candidate_solution)
                    iter_fitnesses.append(fit_val)

                    if fit_val < best_fit_candidate:
                        best_fit_candidate = fit_val
                        best_sol_candidate = candidate_solution
                except Exception as e:
                    logging.error(f"{self.name} Iteration {i+1}, PopMember {pop_idx+1}: Error during fitness evaluation: {e}")
                    continue

            if self.verbose and (i + 1) % (self.n_iterations // 10 or 1) == 0 :
                current_best_in_iter = min(iter_fitnesses) if iter_fitnesses else float('inf')
                logging.info(f"{self.name} Iter {i+1}/{self.n_iterations}: IterBestFit={current_best_in_iter:.4f}, OverallBestFit={best_fit_candidate:.4f}")

        self.solution = best_sol_candidate
        self.fitness = best_fit_candidate
        sol_str = str(self.solution)[:100] + "..." if self.solution and len(str(self.solution)) > 100 else str(self.solution)
        logging.info(f"{self.name} Finished. Best Overall Fitness: {self.fitness:.4f}. Solution (partial): {sol_str}")
        return self.solution, self.fitness

class AquilaOptimizer(PlaceholderNIA): pass
class SlimeMouldAlgorithm(PlaceholderNIA): pass
class ImprovedGreyWolfOptimizer(PlaceholderNIA): pass

# --- UTILITY FUNCTIONS ---
def calculate_ear(eye_landmarks_pts: np.ndarray) -> float:
    if eye_landmarks_pts is None or len(eye_landmarks_pts) != 6: return 0.35
    try:
        A=dist.euclidean(eye_landmarks_pts[1],eye_landmarks_pts[5])
        B=dist.euclidean(eye_landmarks_pts[2],eye_landmarks_pts[4])
        C=dist.euclidean(eye_landmarks_pts[0],eye_landmarks_pts[3])
        return (A+B)/(2.0*C) if C > 1e-6 else 0.35
    except Exception: return 0.35

def get_eye_landmarks_and_center(landmarks_dlib, eye_start_idx, eye_end_idx) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        eye_pts = np.array([(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y)
                            for i in range(eye_start_idx, eye_end_idx)], dtype=np.float32)
        return eye_pts, np.mean(eye_pts, axis=0)
    except Exception: return None, None

def map_detailed_state_to_emotion(detailed_state: str) -> str:
    return STATE_TO_EMOTION_MAP_V2_5.get(detailed_state, DEFAULT_EMOTION_V2_5)

def augment_features_noise(features: np.ndarray, noise_level: float = ML_AUGMENT_NOISE_LEVEL) -> np.ndarray:
    if not isinstance(features, np.ndarray): return features
    return features + np.random.normal(0, noise_level, features.shape)

def augment_features_scaling(features: np.ndarray, scale_range: Tuple[float, float] = ML_AUGMENT_SCALE_RANGE) -> np.ndarray:
    if not isinstance(features, np.ndarray): return features
    return features * random.uniform(scale_range[0], scale_range[1])

# --- V2.5 Validation Metrics ---
def calculate_manual_accuracy(true_labels: List[Any], pred_labels: List[Any]) -> float:
    if not true_labels or len(true_labels) != len(pred_labels): return 0.0
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    return (correct / len(true_labels)) * 100.0

def get_classification_report(true_labels: List[Any], pred_labels: List[Any], class_names: List[str]) -> str:
    report_str = f"Accuracy: {calculate_manual_accuracy(true_labels, pred_labels):.2f}%\n"
    if SKLEARN_AVAILABLE:
        try:
            report_str += "\nscikit-learn Classification Report:\n"
            report_str += classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
            report_str += "\n\nscikit-learn Confusion Matrix:\n"
            cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
            report_str += np.array2string(cm, separator=', ')
        except Exception as e:
            report_str += f"Could not generate sklearn report: {e}\n(This might be due to label mismatches or very few samples per class for averaging)\n"
    else:
        report_str += "(Install scikit-learn for detailed F1 scores and confusion matrix)\n"
    return report_str


# --- EyeFeatureExtractor ---
class EyeFeatureExtractor:
    def __init__(self, fps: float = DEFAULT_PROCESSING_FPS):
        self.fps: float = max(1.0, float(fps))
        saccade_history_len = int(self.fps * 2)
        self.saccade_amplitude_history: Deque[float] = deque(maxlen=max(1, saccade_history_len))
        self.saccade_velocity_history: Deque[float] = deque(maxlen=max(1, saccade_history_len))
        self.gaze_point_history_short_window: Deque[np.ndarray] = deque(maxlen=max(2, FIXATION_STABILITY_WINDOW_FRAMES))
        self.blink_timestamps: Deque[float] = deque(maxlen=int(self.fps * BLINK_RATE_WINDOW_SECONDS * 1.5))
        min_fix_duration_s = MIN_FRAMES_FOR_FIXATION / self.fps if self.fps > 0 else 0.1
        fix_events_approx = 1.0 / (min_fix_duration_s + 0.1) if min_fix_duration_s > 0 else 5
        fix_stats_len = int(fix_events_approx * FIXATION_STATS_WINDOW_SECONDS)
        self.fixation_durations_list: Deque[float] = deque(maxlen=max(10, fix_stats_len))
        self.pupil_proxy_history_norm: Deque[float] = deque(maxlen=max(3, int(self.fps * 0.5)))
        self.model_points_3d: np.ndarray = HEAD_POSE_MODEL_POINTS_3D
        self.camera_matrix: Optional[np.ndarray]=None
        self.last_valid_rotation_vector: Optional[np.ndarray]=None
        self.last_valid_translation_vector: Optional[np.ndarray]=None
        self.reset_state()

    def reset_state(self):
        self.left_ear: float = 0.35; self.right_ear: float = 0.35; self.avg_ear: float = 0.35
        self.pupil_diameter_proxy: float = 0.0; self.inter_ocular_dist_face_ratio: float = 0.0
        self.saccade_amplitude_history.clear(); self.saccade_velocity_history.clear()
        self.gaze_point_history_short_window.clear()
        self.fixation_stability: float = -1.0
        self.head_pose_angles: Tuple[float,float,float] = (0.0,0.0,0.0)
        self.face_bbox_area: float = 0.0; self.face_width: float = 1.0
        self.is_blinking_now: bool = False; self.blink_frame_counter: int = 0; self.total_blinks_session: int = 0
        self.blink_timestamps.clear(); self.blink_rate_bpm: float = 0.0
        self.is_drowsy_closure: bool = False; self.drowsy_frame_counter: int = 0
        self.current_fixation_frames: int = 0; self.fixation_start_timestamp: float = 0.0
        self.fixation_durations_list.clear()
        self.mean_fixation_duration=self.median_fixation_duration=self.max_fixation_duration=self.variance_fixation_duration=0.0
        self.pupil_proxy_history_norm.clear()
        self.pupil_proxy_derivative1=self.pupil_proxy_derivative2=0.0
        self.previous_overall_eye_center: Optional[np.ndarray]=None; self.frame_timestamp: float=0.0

    def _initialize_camera_matrix(self, frame_shape: Tuple[int, int, Any]):
        if len(frame_shape) < 2:
            h, w = 480, 640
        else:
            h, w = frame_shape[0], frame_shape[1]
        fl = float(w)
        cx, cy = float(w/2), float(h/2)
        self.camera_matrix = np.array([[fl,0,cx],[0,fl,cy],[0,0,1]],dtype=np.float64)

    def _estimate_head_pose(self, landmarks_dlib, frame_shape) -> bool:
        if self.camera_matrix is None: self._initialize_camera_matrix(frame_shape)
        if landmarks_dlib is None: return False
        try:
            img_pts_data = [
                (landmarks_dlib.part(NOSE_TIP_IDX).x, landmarks_dlib.part(NOSE_TIP_IDX).y),
                (landmarks_dlib.part(CHIN_IDX).x, landmarks_dlib.part(CHIN_IDX).y),
                (landmarks_dlib.part(L_EYE_CORNER_IDX).x, landmarks_dlib.part(L_EYE_CORNER_IDX).y),
                (landmarks_dlib.part(R_EYE_CORNER_IDX).x, landmarks_dlib.part(R_EYE_CORNER_IDX).y),
                (landmarks_dlib.part(L_MOUTH_CORNER_IDX).x, landmarks_dlib.part(L_MOUTH_CORNER_IDX).y),
                (landmarks_dlib.part(R_MOUTH_CORNER_IDX).x, landmarks_dlib.part(R_MOUTH_CORNER_IDX).y)
            ]
        except Exception: return False

        img_pts = np.array(img_pts_data, dtype=np.float64)
        dist_coeffs=np.zeros((4,1))
        try:
            use_g = self.last_valid_rotation_vector is not None and self.last_valid_translation_vector is not None
            r_g, t_g = (self.last_valid_rotation_vector, self.last_valid_translation_vector) if use_g else (None,None)
            s, r_v, t_v = cv2.solvePnP(self.model_points_3d,img_pts,self.camera_matrix,dist_coeffs,rvec=r_g,tvec=t_g,useExtrinsicGuess=use_g,flags=cv2.SOLVEPNP_ITERATIVE if use_g else cv2.SOLVEPNP_SQPNP)
            if s:
                self.last_valid_rotation_vector,self.last_valid_translation_vector = r_v,t_v
                rm,_=cv2.Rodrigues(r_v); sy=np.sqrt(rm[0,0]**2+rm[1,0]**2)
                sing = sy < 1e-6
                p_rad,y_rad,r_rad = (np.arctan2(-rm[1,2],rm[1,1]), np.arctan2(-rm[2,0],sy), 0.) if sing else \
                                (np.arctan2(rm[2,1],rm[2,2]), np.arctan2(-rm[2,0],sy), np.arctan2(rm[1,0],rm[0,0]))
                self.head_pose_angles = tuple(map(np.degrees, (y_rad,p_rad,r_rad))); return True
        except cv2.error: pass
        except Exception: pass
        return False

    def update(self, landmarks_dlib, face_rect_obj, current_frame_timestamp: float, frame_shape: Tuple[int, int, Any], current_config_thresholds: Dict[str, Any]):
        self.frame_timestamp = current_frame_timestamp
        ear_thresh_blink = float(current_config_thresholds.get('EAR_THRESHOLD_BLINK', DEFAULT_EAR_THRESHOLD_BLINK))
        ear_consec_blink = int(current_config_thresholds.get('EAR_CONSEC_FRAMES_BLINK', DEFAULT_EAR_CONSEC_FRAMES_BLINK))
        ear_thresh_drowsy = float(current_config_thresholds.get('EAR_THRESHOLD_DROWSY', DEFAULT_EAR_THRESHOLD_DROWSY))

        l_pts, l_c = get_eye_landmarks_and_center(landmarks_dlib, L_START_IDX, L_END_IDX)
        r_pts, r_c = get_eye_landmarks_and_center(landmarks_dlib, R_START_IDX, R_END_IDX)

        if l_pts is not None and r_pts is not None:
            self.left_ear,self.right_ear = calculate_ear(l_pts),calculate_ear(r_pts)
            self.avg_ear = (self.left_ear+self.right_ear)/2.0

        if l_c is not None and r_c is not None: self.pupil_diameter_proxy = dist.euclidean(l_c,r_c)

        if face_rect_obj:
            self.face_bbox_area = float(face_rect_obj.width()*face_rect_obj.height())
            self.face_width = float(face_rect_obj.width())
            self.face_width = max(1.0,self.face_width)
            self.inter_ocular_dist_face_ratio = self.pupil_diameter_proxy/self.face_width if self.face_width > 0 else 0.0
        else:
            self.face_bbox_area = 0.0; self.face_width = 1.0; self.inter_ocular_dist_face_ratio = 0.0

        self._estimate_head_pose(landmarks_dlib,frame_shape)

        if self.avg_ear < ear_thresh_blink:
            self.blink_frame_counter+=1
            if self.blink_frame_counter >= ear_consec_blink and not self.is_blinking_now:
                self.is_blinking_now=True; self.total_blinks_session+=1
                self.blink_timestamps.append(self.frame_timestamp)
                self.is_drowsy_closure,self.drowsy_frame_counter=False,0
        else:
            if self.is_blinking_now: self.is_blinking_now=False
            self.blink_frame_counter=0

        if not self.is_blinking_now and self.avg_ear < ear_thresh_drowsy:
            self.drowsy_frame_counter+=1
            if self.drowsy_frame_counter >= EAR_CONSEC_FRAMES_DROWSY and not self.is_drowsy_closure:
                self.is_drowsy_closure=True
        elif self.avg_ear >= ear_thresh_drowsy:
            if self.is_drowsy_closure: self.is_drowsy_closure=False
            self.drowsy_frame_counter=0

        while self.blink_timestamps and self.blink_timestamps[0] < self.frame_timestamp-BLINK_RATE_WINDOW_SECONDS: self.blink_timestamps.popleft()
        self.blink_rate_bpm = (len(self.blink_timestamps)/BLINK_RATE_WINDOW_SECONDS)*60.0 if BLINK_RATE_WINDOW_SECONDS>0 and self.blink_timestamps else 0.0

        if l_c is not None and r_c is not None:
            cur_eye_c = (l_c+r_c)/2.0; self.gaze_point_history_short_window.append(cur_eye_c.copy())
            fix_max_move = int(current_config_thresholds.get('FIXATION_MAX_MOVEMENT_PX', DEFAULT_FIXATION_MAX_MOVEMENT_PX))

            if self.previous_overall_eye_center is not None and not self.is_blinking_now and not self.is_drowsy_closure:
                move = dist.euclidean(cur_eye_c, self.previous_overall_eye_center)
                if move < fix_max_move:
                    if self.current_fixation_frames == 0: self.fixation_start_timestamp = self.frame_timestamp
                    self.current_fixation_frames+=1
                else:
                    if self.current_fixation_frames >= MIN_FRAMES_FOR_FIXATION:
                        dur = self.frame_timestamp - self.fixation_start_timestamp
                        if dur > 1e-3: self.fixation_durations_list.append(dur)
                    self.current_fixation_frames=0; self.saccade_amplitude_history.append(move)
                    if self.fps > 0: self.saccade_velocity_history.append(move*self.fps)
            else:
                if self.current_fixation_frames >= MIN_FRAMES_FOR_FIXATION and self.previous_overall_eye_center is not None:
                    dur = self.frame_timestamp - self.fixation_start_timestamp
                    if dur > 1e-3: self.fixation_durations_list.append(dur)
                self.current_fixation_frames=0
            self.previous_overall_eye_center = cur_eye_c
        else:
            if self.current_fixation_frames >= MIN_FRAMES_FOR_FIXATION and self.previous_overall_eye_center is not None:
                dur = self.frame_timestamp - self.fixation_start_timestamp
                if dur > 1e-3: self.fixation_durations_list.append(dur)
            self.current_fixation_frames = 0

        self.fixation_stability = -1.0
        if self.current_fixation_frames >= FIXATION_STABILITY_WINDOW_FRAMES and len(self.gaze_point_history_short_window) >= max(2, FIXATION_STABILITY_WINDOW_FRAMES//2):
            try:
                gaze_arr = np.array(list(self.gaze_point_history_short_window))
                if gaze_arr.ndim == 2 and gaze_arr.shape[0] >=2 and gaze_arr.shape[1] == 2:
                     self.fixation_stability = np.var(gaze_arr[:,0])+np.var(gaze_arr[:,1])
            except Exception: self.fixation_stability = -1.0

        if self.fixation_durations_list:
            fd_arr=np.array(list(self.fixation_durations_list))
            self.mean_fixation_duration=np.mean(fd_arr); self.median_fixation_duration=np.median(fd_arr)
            self.max_fixation_duration=np.max(fd_arr); self.variance_fixation_duration=np.var(fd_arr) if len(fd_arr)>1 else 0.0
        else: self.mean_fixation_duration=self.median_fixation_duration=self.max_fixation_duration=self.variance_fixation_duration=0.0

        self.pupil_proxy_history_norm.append(self.inter_ocular_dist_face_ratio)
        if len(self.pupil_proxy_history_norm)>=3 and self.fps > 0:
            dt = 1.0 / self.fps
            p0, p1, p2 = self.pupil_proxy_history_norm[-1], self.pupil_proxy_history_norm[-2], self.pupil_proxy_history_norm[-3]
            self.pupil_proxy_derivative1 = (p0 - p1) / dt if dt > 1e-6 else 0.0
            self.pupil_proxy_derivative2 = (p0 - 2*p1 + p2) / (dt*dt) if dt > 1e-6 else 0.0
        else: self.pupil_proxy_derivative1=self.pupil_proxy_derivative2=0.0

    def get_features_dict(self) -> Dict[str, Any]:
        return {"avg_ear":self.avg_ear, "pupil_diameter_proxy":self.pupil_diameter_proxy, "inter_ocular_dist_face_ratio":self.inter_ocular_dist_face_ratio,
                "avg_saccade_amplitude":np.mean(list(self.saccade_amplitude_history)) if self.saccade_amplitude_history else 0.0,
                "avg_saccade_velocity":np.mean(list(self.saccade_velocity_history)) if self.saccade_velocity_history else 0.0,
                "fixation_stability":self.fixation_stability, "head_yaw":self.head_pose_angles[0], "head_pitch":self.head_pose_angles[1], "head_roll":self.head_pose_angles[2],
                "face_bbox_area":self.face_bbox_area, "is_blinking_now":float(self.is_blinking_now), "total_blinks_session":float(self.total_blinks_session),
                "blink_rate_bpm":self.blink_rate_bpm, "is_drowsy_closure":float(self.is_drowsy_closure), "current_fixation_frames":float(self.current_fixation_frames),
                "mean_fixation_duration":self.mean_fixation_duration, "median_fixation_duration":self.median_fixation_duration,
                "max_fixation_duration":self.max_fixation_duration, "variance_fixation_duration":self.variance_fixation_duration,
                "pupil_proxy_derivative1":self.pupil_proxy_derivative1, "pupil_proxy_derivative2":self.pupil_proxy_derivative2}

    def get_numerical_feature_vector(self) -> np.ndarray:
        f = self.get_features_dict()
        fix_stab_val = f["fixation_stability"] if f["fixation_stability"] != -1.0 else DEFAULT_FIXATION_STABILITY_MAX_VARIANCE * 2.5

        vec = np.array([
            f["avg_ear"], f["pupil_diameter_proxy"], f["inter_ocular_dist_face_ratio"],
            f["avg_saccade_amplitude"], f["avg_saccade_velocity"], fix_stab_val,
            f["head_yaw"], f["head_pitch"], f["head_roll"], f["face_bbox_area"],
            f["is_blinking_now"], f["blink_rate_bpm"], f["is_drowsy_closure"],
            f["current_fixation_frames"], f["mean_fixation_duration"], f["median_fixation_duration"],
            f["max_fixation_duration"], f["variance_fixation_duration"],
            f["pupil_proxy_derivative1"], f["pupil_proxy_derivative2"]
        ], dtype=np.float32)
        if len(vec) != ML_INPUT_DIM:
            logging.critical(f"Feature vector length {len(vec)} != ML_INPUT_DIM {ML_INPUT_DIM}. Fix definitions!")
            raise ValueError("Feature vector length mismatch.")
        return vec

# --- EyeStateManager ---
class EyeStateManager:
    def __init__(self, fps: float = DEFAULT_PROCESSING_FPS, initial_thresholds: Optional[Dict[str, Any]] = None):
        self.fps = max(1.0, float(fps))
        self.thresholds: Dict[str, Union[float, int]] = {}
        self.update_thresholds(initial_thresholds if initial_thresholds else {})

        self.T_FIX_FRAMES_FOCUSED_SHORT = int(BASE_DURATION_FIX_FOCUSED_SHORT_S * self.fps)
        self.T_FIX_FRAMES_FOCUSED_INTENSE = int(BASE_DURATION_FIX_FOCUSED_INTENSE_S * self.fps)
        eff_fps_deriv = self.fps
        self.T_IPD_D1_DILATE_STRONG = BASE_T_IPD_D1_DILATE_STRONG_FACTOR * IPD_CHANGE_REL_THRESHOLD_SIGNIFICANT * eff_fps_deriv
        self.T_IPD_D1_CONSTRICT_STRONG = BASE_T_IPD_D1_CONSTRICT_STRONG_FACTOR * IPD_CHANGE_REL_THRESHOLD_SIGNIFICANT * eff_fps_deriv
        self.T_IODR_HIGH_ENGAGE = T_IODR_HIGH_ENGAGE_STATE
        self.T_IODR_LOW_SQUINT = T_IODR_LOW_SQUINT_STATE
        self.T_HEAD_ROLL_THINKING = T_HEAD_ROLL_THINKING_STATE
        self.T_HEAD_YAW_THINKING_MAX = T_HEAD_YAW_THINKING_MAX_STATE
        self.T_SACCADE_AMP_THINKING_MAX = T_SACCADE_AMP_THINKING_MAX_STATE
        self.T_BLINK_RATE_FOCUSED_LOW = T_BLINK_RATE_FOCUSED_LOW_STATE

    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        self.thresholds['EAR_THRESHOLD_BLINK'] = float(new_thresholds.get('EAR_THRESHOLD_BLINK', DEFAULT_EAR_THRESHOLD_BLINK))
        self.thresholds['EAR_CONSEC_FRAMES_BLINK'] = int(new_thresholds.get('EAR_CONSEC_FRAMES_BLINK', DEFAULT_EAR_CONSEC_FRAMES_BLINK))
        self.thresholds['EAR_THRESHOLD_DROWSY'] = float(new_thresholds.get('EAR_THRESHOLD_DROWSY', DEFAULT_EAR_THRESHOLD_DROWSY))
        self.thresholds['FIXATION_MAX_MOVEMENT_PX'] = int(new_thresholds.get('FIXATION_MAX_MOVEMENT_PX', DEFAULT_FIXATION_MAX_MOVEMENT_PX))
        self.thresholds['T_BLINK_RATE_NERVOUS_HIGH_STATE'] = float(new_thresholds.get('T_BLINK_RATE_NERVOUS_HIGH_STATE', DEFAULT_T_BLINK_RATE_NERVOUS_HIGH_STATE))
        base_fix_stab_max_var = float(new_thresholds.get('FIXATION_STABILITY_MAX_VARIANCE', DEFAULT_FIXATION_STABILITY_MAX_VARIANCE))
        fix_stab_good_factor = float(new_thresholds.get('T_FIX_STABILITY_GOOD_STATE_FACTOR', DEFAULT_T_FIX_STABILITY_GOOD_STATE_FACTOR))
        self.thresholds['T_FIX_STABILITY_GOOD'] = base_fix_stab_max_var * fix_stab_good_factor
        self.thresholds['T_FIX_STABILITY_POOR'] = base_fix_stab_max_var * T_FIX_STABILITY_POOR_STATE_FACTOR
        self.thresholds['T_SACCADE_AMP_SCANNING_HIGH_STATE'] = float(new_thresholds.get('T_SACCADE_AMP_SCANNING_HIGH_STATE', DEFAULT_T_SACCADE_AMP_SCANNING_HIGH_STATE))
        self.thresholds['T_EAR_FATIGUE_STATE'] = self.thresholds['EAR_THRESHOLD_DROWSY'] + 0.02

    def _calculate_confidence(self, state: str, conditions_met: int, total_possible_conditions: int) -> float:
        if state in ["Blinking", "Fatigued_Drowsy", "No Face"]: return 0.95
        if total_possible_conditions == 0: return 0.5
        base_confidence = 0.6 + (0.35 * (conditions_met / total_possible_conditions))
        return min(0.95, base_confidence)

    def _check_focused_states(self, features: Dict[str, Any]) -> Tuple[Optional[str], int, int]:
        conditions_met, total_conditions = 0, 4

        is_long_fix = features.get("current_fixation_frames",0.0) >= self.T_FIX_FRAMES_FOCUSED_INTENSE
        is_short_fix = features.get("current_fixation_frames",0.0) >= self.T_FIX_FRAMES_FOCUSED_SHORT
        fix_stab = features.get("fixation_stability", -1.0)
        has_good_stab = (fix_stab != -1.0) and (fix_stab < self.thresholds['T_FIX_STABILITY_GOOD'])
        pupil_dilating = features.get("pupil_proxy_derivative1",0.0) > self.T_IPD_D1_DILATE_STRONG
        iodr_high = features.get("inter_ocular_dist_face_ratio",0.0) > self.T_IODR_HIGH_ENGAGE

        if is_long_fix: conditions_met +=1
        if has_good_stab: conditions_met +=1

        if is_long_fix and has_good_stab:
            specific_cond_met = 0
            if pupil_dilating: specific_cond_met +=1
            if iodr_high: specific_cond_met +=1
            return ("Focused_Intense" if pupil_dilating or iodr_high else "Focused_Deep"), conditions_met + specific_cond_met, total_conditions + 2

        if is_short_fix and has_good_stab:
            specific_cond_met = 0
            if features.get("blink_rate_bpm",0.0) < self.T_BLINK_RATE_FOCUSED_LOW : specific_cond_met +=1
            return ("Focused_Calm" if features.get("blink_rate_bpm",0.0) < self.T_BLINK_RATE_FOCUSED_LOW else "Focused_Attentive"), conditions_met + specific_cond_met, total_conditions + 1
        return None, conditions_met, total_conditions

    def _check_agitated_states(self, features: Dict[str, Any]) -> Tuple[Optional[str], int, int]:
        conditions_met, total_conditions = 0, 5
        is_blink_high = features.get("blink_rate_bpm",0.0) > self.thresholds['T_BLINK_RATE_NERVOUS_HIGH_STATE']
        fix_stab = features.get("fixation_stability", -1.0)
        has_poor_stab = (fix_stab != -1.0) and (fix_stab > self.thresholds['T_FIX_STABILITY_POOR'])
        avg_sacc_amp = features.get("avg_saccade_amplitude",0.0)
        is_jittery = (avg_sacc_amp > 0 and avg_sacc_amp < T_SACCADE_AMP_JITTER_LOW_STATE and
                      features.get("current_fixation_frames",0.0) < (self.T_FIX_FRAMES_FOCUSED_SHORT / 2))
        pupil_constricting = features.get("pupil_proxy_derivative1",0.0) < self.T_IPD_D1_CONSTRICT_STRONG
        iodr_low = features.get("inter_ocular_dist_face_ratio",0.0) < self.T_IODR_LOW_SQUINT

        num_primary_agitated_triggers = 0
        if is_blink_high: num_primary_agitated_triggers +=1
        if has_poor_stab: num_primary_agitated_triggers +=1
        if is_jittery: num_primary_agitated_triggers +=1
        conditions_met += num_primary_agitated_triggers

        if num_primary_agitated_triggers > 0:
            specific_cond_met = 0
            if pupil_constricting: specific_cond_met+=1
            if iodr_low: specific_cond_met+=1
            return ("Nervous_Anxious" if pupil_constricting or iodr_low else "Nervous_Jittery"), conditions_met + specific_cond_met, total_conditions + 2
        return None, conditions_met, total_conditions

    def classify_with_confidence(self, features: Dict[str, Any]) -> Tuple[str, float]:
        if not features: return "No Face", 0.95

        if features.get("is_blinking_now", 0.0) > 0.5 : return "Blinking", 0.98
        if features.get("is_drowsy_closure", 0.0) > 0.5 : return "Fatigued_Drowsy", 0.95

        focused_state, foc_met, foc_total = self._check_focused_states(features)
        if focused_state: return focused_state, self._calculate_confidence(focused_state, foc_met, foc_total)

        agitated_state, agi_met, agi_total = self._check_agitated_states(features)
        if agitated_state: return agitated_state, self._calculate_confidence(agitated_state, agi_met, agi_total)

        conditions_met_scan, total_conditions_scan = 0, 2
        is_scanning = features.get("avg_saccade_amplitude", 0.0) > self.thresholds['T_SACCADE_AMP_SCANNING_HIGH_STATE']
        not_fixating_focused = features.get("current_fixation_frames", 0.0) < (self.T_FIX_FRAMES_FOCUSED_SHORT / 2)
        if is_scanning: conditions_met_scan +=1
        if not_fixating_focused: conditions_met_scan +=1
        if is_scanning and not_fixating_focused:
             return "Distracted_Scanning", self._calculate_confidence("Distracted_Scanning", conditions_met_scan, total_conditions_scan)

        conditions_met_fatigue, total_conditions_fatigue = 0, 2
        avg_ear_val = features.get("avg_ear", 0.35)
        is_ear_lowish = (avg_ear_val < self.thresholds['T_EAR_FATIGUE_STATE'] and
                         avg_ear_val > self.thresholds['EAR_THRESHOLD_DROWSY'])
        mod_high_blink = features.get("blink_rate_bpm",0.0) > (self.thresholds['T_BLINK_RATE_NERVOUS_HIGH_STATE'] / 1.5)
        if is_ear_lowish: conditions_met_fatigue +=1
        if mod_high_blink: conditions_met_fatigue +=1
        if is_ear_lowish and mod_high_blink:
            return "Fatigued_Subtle", self._calculate_confidence("Fatigued_Subtle", conditions_met_fatigue, total_conditions_fatigue)

        conditions_met_think, total_conditions_think = 0, 4
        head_roll_sig = abs(features.get("head_roll",0.0)) > self.T_HEAD_ROLL_THINKING
        head_yaw_mod = abs(features.get("head_yaw",0.0)) < self.T_HEAD_YAW_THINKING_MAX
        sacc_small = features.get("avg_saccade_amplitude",0.0) < self.T_SACCADE_AMP_THINKING_MAX
        min_fix_think = MIN_FRAMES_FOR_FIXATION / 2 if MIN_FRAMES_FOR_FIXATION > 0 else 1
        has_min_fix_think = features.get("current_fixation_frames",0.0) > min_fix_think
        if head_roll_sig: conditions_met_think +=1
        if head_yaw_mod: conditions_met_think +=1
        if sacc_small: conditions_met_think +=1
        if has_min_fix_think: conditions_met_think +=1
        if head_roll_sig and head_yaw_mod and sacc_small and has_min_fix_think:
            return "Thinking_Pondering", self._calculate_confidence("Thinking_Pondering", conditions_met_think, total_conditions_think)

        return "Natural_Neutral", 0.5

    def determine_overall_condition(self, detailed_state_counts: Counter, total_frames: int) -> Tuple[str, str]:
        if not detailed_state_counts or total_frames == 0: return "Undetermined", "No eye state data."
        emotion_counts = Counter()
        for state, count in detailed_state_counts.items(): emotion_counts[map_detailed_state_to_emotion(state)] += count
        if not emotion_counts: return "Undetermined (Mapping)", "Could not map states."

        significant_emotions = Counter({
            emo: ct for emo, ct in emotion_counts.items()
            if emo not in ["No Face/Unknown", "Blinking"] or \
               (emo == "Blinking" and (ct / total_frames) > 0.5)
        })

        if not significant_emotions:
            if "Blinking" in emotion_counts and (emotion_counts["Blinking"] / total_frames) > 0.2:
                most_common_emo, emo_count = "Blinking", emotion_counts["Blinking"]
            elif "No Face/Unknown" in emotion_counts:
                most_common_emo, emo_count = "No Face/Unknown", emotion_counts["No Face/Unknown"]
            else:
                most_common_emo, emo_count = emotion_counts.most_common(1)[0]
        else:
            most_common_emo, emo_count = significant_emotions.most_common(1)[0]

        perc = (emo_count/total_frames)*100
        expl = f"Overall Emotion: '{most_common_emo}' ({perc:.1f}% of frames). "
        if "Tired/Fatigued" == most_common_emo: expl += "Suggests significant tiredness."
        elif "Engaged/Focused" == most_common_emo: expl += "Indicates deep concentration."
        elif "Agitated/Stressed" == most_common_emo: expl += "Suggests stress or nervousness."
        elif "Distracted/Scanning" == most_common_emo: expl += "Indicates distraction."
        elif "Neutral/Calm" == most_common_emo and perc > 40 : expl += "Predominantly calm or neutral behavior."
        elif "Pondering" == most_common_emo and perc > 15 : expl += "Significant time spent in a thoughtful state."
        elif "Blinking" == most_common_emo and perc > 20: expl += "Notable blinking frequency."
        elif "No Face/Unknown" == most_common_emo: expl += "Face/eyes often not detected."
        else: expl += "General assessment."
        return most_common_emo, expl

# --- V2.5 Hybrid Model Fusion Logic ---
def hybrid_fusion_strategy(
    rule_pred: str, rule_conf: float,
    ml_pred_probs: np.ndarray,
    ml_class_idx_to_name: Callable[[int], str],
    ml_confidence_threshold: float = HYBRID_CONFIDENCE_THRESHOLD_ML,
    rule_confidence_threshold: float = HYBRID_CONFIDENCE_THRESHOLD_RULE
) -> Tuple[str, float, str]:

    ml_top_class_idx = np.argmax(ml_pred_probs)
    ml_top_confidence = ml_pred_probs[ml_top_class_idx]
    ml_pred_name = ml_class_idx_to_name(ml_top_class_idx)

    if ml_top_confidence >= ml_confidence_threshold and ml_pred_name not in ["Blinking", "No Face/Unknown"]:
        return ml_pred_name, ml_top_confidence, "ML"

    if rule_conf >= rule_confidence_threshold and rule_pred not in ["Blinking", "No Face/Unknown"]:
        return rule_pred, rule_conf, "Rule"

    if ml_pred_name in ["Blinking", "No Face/Unknown"] and ml_top_confidence > 0.6:
        if rule_pred == ml_pred_name or rule_conf < 0.5:
             return ml_pred_name, ml_top_confidence, "ML (Aux)"

    if rule_pred in ["Blinking", "No Face/Unknown"] and rule_conf > 0.8:
        return rule_pred, rule_conf, "Rule (Aux)"

    if ml_pred_name in ["Blinking", "No Face/Unknown"] and rule_pred not in ["Blinking", "No Face/Unknown"] and rule_conf > 0.4:
        return rule_pred, rule_conf, "Hybrid (Rule Preferred)"

    if rule_pred in ["Blinking", "No Face/Unknown"] and ml_pred_name not in ["Blinking", "No Face/Unknown"] and ml_top_confidence > 0.4:
        return ml_pred_name, ml_top_confidence, "Hybrid (ML Preferred)"

    if rule_conf >= ml_top_confidence:
        return rule_pred, rule_conf, "Hybrid (Default Rule)"
    else:
        return ml_pred_name, ml_top_confidence, "Hybrid (Default ML)"

# --- Visualization & PDF (ReportLab V2.5) ---
def draw_diagnostics_on_frame_v2_5(frame: np.ndarray, features_dict: Dict[str, Any],
                                   display_emotion: str, display_state: str, display_confidence: float, display_source: str,
                                   face_rect_tuple: Optional[Tuple[int,int,int,int]],
                                   landmarks, overall_disp: Optional[str],
                                   raw_rule_info: Optional[str]=None, raw_ml_info: Optional[str]=None) -> None:
    y, font, ssm, slg, sem = 20, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0.6, 0.7
    c_txt,c_st,c_em,c_ovr, c_raw = (220,220,100),(50,50,255),(50,255,50),(255,255,0), (150,150,150)
    th,th_lg = 1,2; line_h=18; y_start = y

    cv2.putText(frame,f"Emotion ({display_source}): {display_emotion} ({display_confidence:.2f})",(10,y),font,sem,c_em,th_lg); y+=25
    cv2.putText(frame,f"State: {display_state}",(10,y),font,slg,c_st,th_lg); y+=25

    if raw_rule_info: cv2.putText(frame, f"Rule: {raw_rule_info}", (10,y), font, ssm-0.05, c_raw, th); y+=18
    if raw_ml_info: cv2.putText(frame, f"ML: {raw_ml_info}", (10,y), font, ssm-0.05, c_raw, th); y+=20

    diag_items = list({
        "EAR": f"{features_dict.get('avg_ear', 0.):.2f}",
        "Blinks": f"{int(features_dict.get('total_blinks_session',0))} ({features_dict.get('blink_rate_bpm',0.):.1f} BPM)",
        "Drowsy": "Y" if features_dict.get("is_drowsy_closure",0.0) > 0.5 else "N",
        "IOD Ratio":f"{features_dict.get('inter_ocular_dist_face_ratio',0.):.3f}",
        "Fix Frames":f"{int(features_dict.get('current_fixation_frames',0))}",
        "Head Roll":f"{features_dict.get('head_roll',0.):.1f}°"
    }.items())
    for i, (k,v) in enumerate(diag_items): cv2.putText(frame,f"{k}: {v}",(10,y+i*line_h),font,ssm,c_txt,th)
    if overall_disp: cv2.putText(frame,f"Overall: {overall_disp}",(10,frame.shape[0]-15),font,slg,c_ovr,th_lg)
    if face_rect_tuple: l,t,r,b=face_rect_tuple; cv2.rectangle(frame,(l,t),(r,b),(0,255,0),1)
    if landmarks:
        for i in range(L_START_IDX,R_END_IDX):
            try: pt=landmarks.part(i); cv2.circle(frame,(pt.x,pt.y),1,(0,255,255),-1)
            except: pass

def generate_pdf_report_reportlab_v2_5(output_filename: str, in_vid_name: str, report_data: Dict):
    doc = SimpleDocTemplate(output_filename)
    story: List[Any] = []
    styles = getSampleStyleSheet()

    registered_fonts = {'normal': 'Helvetica', 'bold': 'Helvetica-Bold', 'italic': 'Helvetica-Oblique'}
    try:
        if os.path.exists(DEJAVU_FONT_PATH):
            pdfmetrics.registerFont(TTFont('DejaVuSans', DEJAVU_FONT_PATH))
            registered_fonts['normal'] = 'DejaVuSans'
            if os.path.exists(DEJAVU_FONT_BOLD_PATH):
                pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', DEJAVU_FONT_BOLD_PATH))
                registered_fonts['bold'] = 'DejaVuSans-Bold'
            if os.path.exists(DEJAVU_FONT_ITALIC_PATH):
                pdfmetrics.registerFont(TTFont('DejaVuSans-Italic', DEJAVU_FONT_ITALIC_PATH))
                registered_fonts['italic'] = 'DejaVuSans-Italic'
        else:
            logging.warning(f"DejaVuSans font not found at {DEJAVU_FONT_PATH}. PDF will use Helvetica fallback.")

    except Exception as e: logging.error(f"PDF Font Registration Error: {e}. Using Helvetica fallback.")

    title_style = ParagraphStyle('TitleStyle', fontName=registered_fonts['bold'], fontSize=16, alignment=TA_CENTER, spaceAfter=0.2*inch)
    header_style = ParagraphStyle('HeaderStyle', fontName=registered_fonts['bold'], fontSize=13, spaceBefore=0.2*inch, spaceAfter=0.1*inch)
    body_style = ParagraphStyle('BodyStyle', fontName=registered_fonts['normal'], fontSize=10, leading=12, spaceAfter=6)
    italic_body_style = ParagraphStyle('ItalicBodyStyle', fontName=registered_fonts['italic'], fontSize=10, leading=12, spaceAfter=6)
    small_body_style = ParagraphStyle('SmallBodyStyle', fontName=registered_fonts['normal'], fontSize=9, leading=10, spaceAfter=4)
    code_style = ParagraphStyle('CodeStyle', fontName='Courier', fontSize=8, leading=10)

    story.append(Paragraph(f"Eye Behavior Analysis Report ({SCRIPT_VERSION})", title_style))
    story.append(Paragraph(f"<b>Video Source:</b> {in_vid_name}", body_style))
    proc_sum = report_data.get("processing_summary", {})
    story.append(Paragraph(f"<b>Frames Processed:</b> {proc_sum.get('processed_frames','N/A')} / {proc_sum.get('total_input_frames','N/A')} (Input)", body_style))
    story.append(Paragraph(f"<b>Avg. Processing FPS:</b> {proc_sum.get('avg_processing_fps',0.0):.1f}", body_style))
    story.append(Paragraph(f"<b>Temporal Smoothing (Rules):</b> {proc_sum.get('smoothing_window', 'N/A')}", body_style))
    story.append(Spacer(1,0.1*inch))

    overall_sum = report_data.get("overall_summary", {})
    story.append(Paragraph("Overall Session Summary (Primary System):", header_style))
    story.append(Paragraph(f"<b>Predominant Emotion:</b> {overall_sum.get('emotion', 'N/A')}", body_style))
    story.append(Paragraph(f"<i>Interpretation:</i> {overall_sum.get('explanation', 'N/A')}", italic_body_style))
    story.append(Spacer(1,0.1*inch))

    story.append(Paragraph("Emotion Distribution (Primary System):", header_style))
    emo_dist = report_data.get("emotion_distribution", Counter())
    tot_frames = proc_sum.get('processed_frames', 1)
    if tot_frames > 0 and emo_dist:
        for emo, ct in emo_dist.most_common():
            story.append(Paragraph(f"• {emo}: {ct} frames ({(ct/tot_frames)*100 :.1f}%)", body_style))
    else:
        story.append(Paragraph("No emotion distribution data.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    avg_fs = report_data.get("average_features", {})
    if avg_fs:
        story.append(Paragraph("Average Feature Values (Session Summary):",header_style))
        feat_data_table_content = []
        feat_items_for_report = {
            "Avg. EAR": avg_fs.get('avg_avg_ear'), "Avg. Blink Rate (BPM)": avg_fs.get('avg_blink_rate_bpm'),
            "Avg. Saccade Amp. (px)": avg_fs.get('avg_avg_saccade_amplitude'), "Avg. Fixation Dur. (s)": avg_fs.get('avg_mean_fixation_duration'),
        }
        valid_feat_items = [(k,v) for k,v in feat_items_for_report.items() if v is not None]
        if valid_feat_items:
            for lbl, val in valid_feat_items:
                  v_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                  feat_data_table_content.append([Paragraph(lbl, small_body_style), Paragraph(v_str, small_body_style)])
            feat_table = Table(feat_data_table_content, colWidths=[3*inch,3*inch])
            feat_table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
            story.append(feat_table)
        story.append(Spacer(1, 0.1 * inch))

    val_report = report_data.get("validation_report_rule_based")
    if val_report:
        story.append(PageBreak())
        story.append(Paragraph("Rule-Based System - Preliminary Validation:", header_style))
        story.append(Paragraph(f"Test Data Source: {val_report.get('source', 'N/A')}", small_body_style))
        story.append(Paragraph(f"Number of Test Samples: {val_report.get('num_samples', 'N/A')}", small_body_style))
        story.append(Spacer(1,0.1*inch))
        report_text = val_report.get('report_str', 'No detailed report string.')
        for line in report_text.split('\n'):
            story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        story.append(Spacer(1,0.1*inch))

    ml_val_report = report_data.get("validation_report_ml")
    if ml_val_report:
        story.append(Paragraph("ML Model - Preliminary Validation:", header_style))
        story.append(Paragraph(f"Test Data Source: {ml_val_report.get('source', 'N/A')}", small_body_style))
        story.append(Paragraph(f"Number of Test Samples: {ml_val_report.get('num_samples', 'N/A')}", small_body_style))
        story.append(Spacer(1,0.1*inch))
        report_text_ml = ml_val_report.get('report_str', 'No detailed report string.')
        for line in report_text_ml.split('\n'):
            story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        story.append(Spacer(1,0.1*inch))

    def footer_func(canvas, doc):
        canvas.saveState()
        canvas.setFont(registered_fonts['italic'], 8)
        canvas.setFillColor(colors.grey)
        gen_time_str = f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} ({SCRIPT_VERSION})"
        try:
            text_width = pdfmetrics.stringWidth(gen_time_str, registered_fonts['italic'], 8)
            canvas.drawString((doc.width + 2*doc.leftMargin - text_width) / 2.0 , 0.5 * inch, gen_time_str)
        except Exception as e:
            logging.warning(f"PDF footer font error: {e}. Footer might be misaligned or use fallback.")
            canvas.drawString(doc.leftMargin, 0.5 * inch, gen_time_str)
        canvas.restoreState()

    try:
        doc.build(story, onFirstPage=footer_func, onLaterPages=footer_func)
        logging.info(f"PDF {SCRIPT_VERSION} Report Saved: {output_filename}")
    except Exception as e:
        logging.error(f"PDF {SCRIPT_VERSION} Save Error: {e}", exc_info=True)

# --- ML Model & Dataset ---
class EyeEmotionLSTM(nn.Module):
    def __init__(self, input_dim=ML_INPUT_DIM, hidden_dim=ML_HIDDEN_DIM_LSTM,
                 num_layers=ML_NUM_LAYERS_LSTM, num_classes=NUM_ML_EMOTION_CLASSES_V2_5,
                 dropout_rate=ML_DROPOUT_RATE_MODEL):
        super().__init__()
        if input_dim <= 0: raise ValueError("Input dim for LSTM must be > 0.")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:,-1,:]))

class EyeFeatureDataset(Dataset):
    def __init__(self, feat_seqs: List[np.ndarray], labels: List[int],
                 seq_len: int = ML_SEQ_LENGTH, in_dim: int = ML_INPUT_DIM,
                 augment: bool = False, noise_level: float = ML_AUGMENT_NOISE_LEVEL,
                 scale_range: Tuple[float, float] = ML_AUGMENT_SCALE_RANGE):
        if len(feat_seqs) != len(labels): raise ValueError("Seq/label len mismatch")
        self.feat_seqs, self.labels = feat_seqs, labels
        self.seq_len, self.in_dim = seq_len, max(1, in_dim)
        self.augment, self.noise_level, self.scale_range = augment, noise_level, scale_range

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_np = np.array(self.feat_seqs[idx], dtype=np.float32)
        if seq_np.ndim == 1 and seq_np.shape[0] == self.in_dim: seq_np = seq_np.reshape(1, self.in_dim)
        elif seq_np.ndim != 2 or seq_np.shape[1] != self.in_dim:
            seq_np = np.zeros((1, self.in_dim), dtype=np.float32)
        if seq_np.shape[0] == 0: seq_np = np.zeros((1, self.in_dim), dtype=np.float32)

        orig_len = seq_np.shape[0]
        if self.augment:
            aug_parts = [augment_features_scaling(augment_features_noise(seq_np[i,:],self.noise_level),self.scale_range) for i in range(orig_len)]
            if aug_parts: seq_np = np.array(aug_parts, dtype=np.float32)

        if orig_len < self.seq_len:
            pad = np.zeros((self.seq_len - orig_len, self.in_dim), dtype=np.float32)
            proc_seq_np = np.concatenate((pad, seq_np), axis=0)
        else: proc_seq_np = seq_np[-self.seq_len:, :]

        return torch.tensor(proc_seq_np, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- NIA Integration Components & PoC Training (Simulated offline tuning) ---
TUNABLE_THRESHOLDS_RULE_BASED = {
    'EAR_THRESHOLD_BLINK': (0.15, 0.30), 'EAR_CONSEC_FRAMES_BLINK': (1, 5),
    'EAR_THRESHOLD_DROWSY': (0.10, 0.20), 'FIXATION_MAX_MOVEMENT_PX': (5, 15),
    'T_BLINK_RATE_NERVOUS_HIGH_STATE': (25.0, 45.0),
    'FIXATION_STABILITY_MAX_VARIANCE': (30.0, 100.0),
    'T_FIX_STABILITY_GOOD_STATE_FACTOR': (0.4, 0.9),
    'T_SACCADE_AMP_SCANNING_HIGH_STATE': (15.0, 40.0)
}
TUNABLE_THRESHOLD_KEYS_ORDERED = list(TUNABLE_THRESHOLDS_RULE_BASED.keys())
DUMMY_ANNOTATED_CLIPS_FOR_NIA = [
    ("c1_focused","Engaged/Focused"), ("c2_distracted","Distracted/Scanning"),
    ("c3_tired","Tired/Fatigued"), ("c4_calm","Neutral/Calm"),
    ("c5_stressed","Agitated/Stressed"), ("c6_pondering", "Pondering")
]

def generate_dummy_ml_data(num_samples: int, seq_length: int, input_dim: int, num_classes: int, is_test_set: bool=False) -> Tuple[List[np.ndarray], List[int], List[str]]:
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []

    if num_samples <=0 or seq_length <=0 or input_dim <=0 or num_classes <=0:
        logging.warning("Invalid params for dummy data. Returning empty.")
        return sequences, labels, sample_ids

    for i in range(num_samples):
        sample_ids.append(f"{'test' if is_test_set else 'train'}_sample_{i}")
        current_seq_len = random.randint(max(1, seq_length // 2), int(seq_length * 1.2))
        current_seq_len = max(1, current_seq_len)
        sequence = np.random.rand(current_seq_len, input_dim).astype(np.float32) * 0.1
        if input_dim > 0: sequence[:, 0] = np.clip(np.random.normal(0.25, 0.08, current_seq_len), 0.05, 0.45)
        if input_dim > 11: sequence[:, 11] = np.clip(np.random.normal(15, 10, current_seq_len), 0, 50)
        label = random.randint(0, num_classes - 1)
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels, sample_ids


def fitness_function_rule_based(threshold_values: List[float], state_mgr_template: EyeStateManager,
                                 dummy_clips_data: List[Dict]) -> float:
    cand_thresh = {k: (round(threshold_values[i]) if k in ['EAR_CONSEC_FRAMES_BLINK','FIXATION_MAX_MOVEMENT_PX'] else threshold_values[i])
                   for i,k in enumerate(TUNABLE_THRESHOLD_KEYS_ORDERED)}
    try:
        state_mgr_template.update_thresholds(cand_thresh)
    except Exception as e:
        logging.error(f"Error updating thresholds in fitness_function_rule_based: {e}")
        return float('inf')

    correct = 0; total = len(dummy_clips_data)
    if total == 0: return float('inf')
    for clip_data in dummy_clips_data:
        sim_feats = clip_data["features"]; gt_emotion = clip_data["ground_truth_emotion"]
        pred_state, _ = state_mgr_template.classify_with_confidence(sim_feats)
        pred_emotion = map_detailed_state_to_emotion(pred_state)
        if pred_emotion == gt_emotion: correct += 1
    return 1.0 - (correct / total)

def run_nia_for_rule_based_tuning(state_manager_to_tune: EyeStateManager) -> Optional[Dict[str, Any]]:
    logging.info(f"--- {SCRIPT_VERSION}: NIA Rule-Based Threshold Tuning ---")
    sim_clip_data = []
    for clip_id, gt_emo in DUMMY_ANNOTATED_CLIPS_FOR_NIA:
        df = {'avg_ear': random.uniform(0.1,0.4), 'blink_rate_bpm': random.uniform(3,35),
              'current_fixation_frames': float(random.randint(0, int(DEFAULT_PROCESSING_FPS*2))),
              'is_blinking_now':0.0, 'is_drowsy_closure':0.0,
              'fixation_stability': random.uniform(5,100),
              'avg_saccade_amplitude': random.uniform(1,30),
              'inter_ocular_dist_face_ratio': random.uniform(0.1,0.2),
              'pupil_proxy_derivative1': random.uniform(-0.05, 0.05) * DEFAULT_PROCESSING_FPS,
              'head_roll': random.uniform(-15,15), 'head_yaw': random.uniform(-15,15),
              }
        if gt_emo == "Engaged/Focused": df['avg_saccade_amplitude'] = random.uniform(1,5)
        elif gt_emo == "Tired/Fatigued": df['avg_ear'] = random.uniform(0.1,0.18)
        sim_clip_data.append({"clip_id": clip_id, "features": df, "ground_truth_emotion": gt_emo})

    if not sim_clip_data: logging.warning("No sim data for NIA tuning."); return None
    lb = [TUNABLE_THRESHOLDS_RULE_BASED[k][0] for k in TUNABLE_THRESHOLD_KEYS_ORDERED]
    ub = [TUNABLE_THRESHOLDS_RULE_BASED[k][1] for k in TUNABLE_THRESHOLD_KEYS_ORDERED]
    prob = {"fit_func": lambda s: fitness_function_rule_based(s, state_manager_to_tune, sim_clip_data), "lb":lb, "ub":ub}

    nia_opt = AquilaOptimizer(pop_size=15, n_iterations=20, problem_dict=prob, verbose=True)
    best_sol, best_fit = nia_opt.solve()
    if best_sol:
        opt_thresh = {k:(round(best_sol[i]) if k in ['EAR_CONSEC_FRAMES_BLINK','FIXATION_MAX_MOVEMENT_PX'] else best_sol[i]) for i,k in enumerate(TUNABLE_THRESHOLD_KEYS_ORDERED)}
        logging.info(f"NIA Rule Opt Thresh: {opt_thresh}, Fitness (1-Acc): {best_fit:.4f}"); return opt_thresh
    logging.warning("NIA Rule Tuning: No solution."); return None

def run_nia_for_ml_feature_selection(augment_in_nia: bool) -> Optional[List[int]]:
    logging.info(f"--- {SCRIPT_VERSION}: NIA ML Feature Selection PoC (SMA) ---")
    logging.warning("NIA ML feature selection is a placeholder. Skipping.")
    # In a real scenario, this would load pre-computed selected features from a file.
    return None

def run_nia_for_ml_hyperparameter_tuning(augment_in_nia: bool, selected_indices: Optional[List[int]]=None) -> Optional[Dict[str,Any]]:
    logging.info(f"--- {SCRIPT_VERSION}: NIA ML Hyperparameter Tuning PoC (IGWO) ---")
    logging.warning("NIA ML hyperparameter tuning is a placeholder. Skipping.")
    # In a real scenario, this would load pre-computed hyperparameters from a file.
    return None

# --- ML Model Training Epoch (Helper, for offline use or testing) ---
def train_ml_model_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                           optimizer: optim.Optimizer, device: torch.device,
                           is_train: bool, epoch_tag: str = "") -> Tuple[float, float]:
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    if len(dataloader) == 0:
        logging.warning(f"{epoch_tag}: Dataloader is empty. Skipping {'train' if is_train else 'eval'} epoch.")
        return 0.0, 0.0

    with torch.set_grad_enabled(is_train):
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if is_train: optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            except Exception as e:
                logging.error(f"Error in {epoch_tag} batch {batch_idx}: {e}", exc_info=True)
                if is_train: optimizer.zero_grad()
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return avg_loss, accuracy

# --- V2.5 Preliminary Validation Functions (For testing, not live inference) ---
def validate_rule_based_system(state_manager: EyeStateManager, test_data: List[Dict[str, Any]], test_data_source_name: str) -> Dict:
    logging.info(f"--- {SCRIPT_VERSION}: Validating Rule-Based System on '{test_data_source_name}' ---")
    true_labels, pred_labels = [], []
    if not test_data:
        logging.warning("No test data provided for rule-based validation.")
        return {"source": test_data_source_name, "num_samples": 0, "report_str": "No test data."}

    for sample in test_data:
        features = sample.get("features")
        gt_emotion = sample.get("ground_truth_emotion")
        if features is None or gt_emotion is None:
            logging.warning(f"Skipping test sample due to missing features or ground truth: {sample.get('id', 'Unknown ID')}")
            continue

        pred_state, _ = state_manager.classify_with_confidence(features)
        pred_emotion = map_detailed_state_to_emotion(pred_state)
        true_labels.append(gt_emotion)
        pred_labels.append(pred_emotion)

    report_str = "Validation not performed (no valid samples)."
    if true_labels:
        report_str = get_classification_report(true_labels, pred_labels, EMOTION_CLASSES_V2_5)
        logging.info(f"Rule-Based Validation on '{test_data_source_name}' Results:\n{report_str}")

    return {"source": test_data_source_name, "num_samples": len(true_labels), "report_str": report_str}

def validate_ml_model(
    model: Optional[EyeEmotionLSTM],
    test_feature_sequences: List[np.ndarray],
    test_labels: List[int],
    test_data_source_name: str,
    device: torch.device,
    input_dim_for_model: int,
    sequence_length: int = ML_SEQ_LENGTH,
    batch_size: int = ML_BATCH_SIZE
) -> Dict:
    logging.info(f"--- {SCRIPT_VERSION}: Validating ML Model on '{test_data_source_name}' ---")
    if model is None:
        logging.warning("No ML model provided for validation.")
        return {"source": test_data_source_name, "num_samples": len(test_labels), "report_str": "ML Model not available."}
    if not test_feature_sequences or not test_labels:
        logging.warning("No test data provided for ML validation.")
        return {"source": test_data_source_name, "num_samples": 0, "report_str": "No test data for ML model."}

    model.eval()
    test_dataset = EyeFeatureDataset(test_feature_sequences, test_labels, seq_len=sequence_length, in_dim=input_dim_for_model, augment=False)
    if len(test_dataset) == 0:
        logging.warning("Test dataset for ML validation is empty.")
        return {"source": test_data_source_name, "num_samples": 0, "report_str": "Test dataset empty."}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_pred_labels, all_true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs.data, 1)
            all_pred_labels.extend([EMOTION_CLASSES_V2_5[idx.item()] for idx in predicted_indices])
            all_true_labels.extend([EMOTION_CLASSES_V2_5[idx.item()] for idx in labels])

    report_str = "Validation not performed (no predictions generated)."
    if all_true_labels:
        report_str = get_classification_report(all_true_labels, all_pred_labels, EMOTION_CLASSES_V2_5)
        logging.info(f"ML Model Validation on '{test_data_source_name}' Results:\n{report_str}")

    return {"source": test_data_source_name, "num_samples": len(all_true_labels), "report_str": report_str}

# This function is meant for OFFLINE training, not to be called in a web request
def run_dummy_ml_training_poc(augment_data: bool,
                               selected_feat_indices: Optional[List[int]] = None,
                               tuned_ml_hyperparams: Optional[Dict] = None) -> Optional[EyeEmotionLSTM]:
    logging.info(f"--- {SCRIPT_VERSION}: Basic ML Model Training PoC (Augment: {augment_data}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"ML training on device: {device}")

    current_model_input_dim = len(selected_feat_indices) if selected_feat_indices else ML_INPUT_DIM
    if current_model_input_dim <= 0:
        logging.error("Cannot train ML model with 0 input features."); return None

    all_train_seqs, train_labels, _ = generate_dummy_ml_data(ML_DUMMY_NUM_TRAIN_SAMPLES, ML_SEQ_LENGTH, ML_INPUT_DIM, NUM_ML_EMOTION_CLASSES_V2_5)
    all_val_seqs, val_labels, _ = generate_dummy_ml_data(ML_DUMMY_NUM_VAL_SAMPLES, ML_SEQ_LENGTH, ML_INPUT_DIM, NUM_ML_EMOTION_CLASSES_V2_5)

    train_seqs_final, val_seqs_final = all_train_seqs, all_val_seqs
    train_labels_final, val_labels_final = train_labels, val_labels
    if selected_feat_indices:
        train_seqs_final = [s[:, selected_feat_indices] for s in all_train_seqs if s.ndim==2 and s.shape[1]>=len(selected_feat_indices)]
        val_seqs_final   = [s[:, selected_feat_indices] for s in all_val_seqs if s.ndim==2 and s.shape[1]>=len(selected_feat_indices)]

    if not train_seqs_final or not val_seqs_final: logging.error("Not enough data after FS for ML PoC."); return None

    train_dataset = EyeFeatureDataset(train_seqs_final, train_labels_final, seq_len=ML_SEQ_LENGTH, in_dim=current_model_input_dim, augment=augment_data)
    val_dataset = EyeFeatureDataset(val_seqs_final, val_labels_final, seq_len=ML_SEQ_LENGTH, in_dim=current_model_input_dim, augment=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0: logging.error("Dataset empty for ML PoC."); return None

    train_loader = DataLoader(train_dataset, ML_BATCH_SIZE, shuffle=True, drop_last=(len(train_dataset) >= ML_BATCH_SIZE))
    val_loader = DataLoader(val_dataset, ML_BATCH_SIZE, shuffle=False, drop_last=(len(val_dataset) >= ML_BATCH_SIZE))
    if len(train_loader) == 0 or len(val_loader) == 0: logging.error("DataLoader empty for ML PoC."); return None

    hp = tuned_ml_hyperparams or {}
    model = EyeEmotionLSTM(input_dim=current_model_input_dim, hidden_dim=hp.get('hidden_dim',ML_HIDDEN_DIM_LSTM),
                           num_layers=hp.get('num_layers',ML_NUM_LAYERS_LSTM), num_classes=NUM_ML_EMOTION_CLASSES_V2_5,
                           dropout_rate=hp.get('dropout_rate',ML_DROPOUT_RATE_MODEL)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hp.get('learning_rate',ML_LEARNING_RATE), weight_decay=ML_OPTIMIZER_WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(ML_NUM_EPOCHS_POC):
        ep_start = time.time()
        tr_loss, tr_acc = train_ml_model_epoch(model, train_loader, criterion, optimizer, device, True, f"TrainPOC Ep{epoch+1}")
        val_loss, val_acc = train_ml_model_epoch(model, val_loader, criterion, optimizer, device, False, f"ValPOC Ep{epoch+1}")
        logging.info(f"Ep {epoch+1}/{ML_NUM_EPOCHS_POC} |Tr L:{tr_loss:.3f} A:{tr_acc:.1f}% |Val L:{val_loss:.3f} A:{val_acc:.1f}% |T:{time.time()-ep_start:.1f}s")
        if val_acc > best_val_acc: best_val_acc = val_acc

    logging.info(f"Dummy ML PoC training finished. Best Val Acc: {best_val_acc:.1f}%")
    # This model would typically be saved to disk here (e.g., torch.save(model.state_dict(), "path/to/model.pt"))
    return model



########################### eye_analysis class ##############################

'''# class EyeAnalysisView(View):

#     # Configuration for this run (controlled by class attributes, not runtime flags from main block)
#     APPLY_RULE_TEMPORAL_SMOOTHING = True
#     RUN_HYBRID_MODEL_LOGIC = True
#     RUN_PRELIMINARY_VALIDATION = True # This will add latency, for demo purposes
#     RUN_VIDEO_PROCESSING = True # Enable video processing logic

#     def __init__(self):
#         super().__init__()
#         self.detector = None
#         self.predictor = None
#         self.ml_model_instance = None
#         self.ml_model_selected_features = None # Assume pre-selected features if any
#         self.optimized_rules = None # Assume pre-optimized rules if any
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # --- Load Dlib Models ---
#         shape_predictor_full_path = '/home/student/new_api/testpro-main/models/shape_predictor_68_face_landmarks.dat'
#         if not os.path.exists(shape_predictor_full_path):
#             logging.critical(f"Dlib model '{SHAPE_PREDICTOR_FILENAME}' NOT FOUND at: {shape_predictor_full_path}.")
#             self.dlib_loaded = False
#         else:
#             try:
#                 self.detector = dlib.get_frontal_face_detector()
#                 self.predictor = dlib.shape_predictor(shape_predictor_full_path)
#                 self.dlib_loaded = True
#                 logging.info("Dlib models loaded successfully.")
#             except Exception as e:
#                 logging.critical(f"Error loading Dlib models: {e}")
#                 self.dlib_loaded = False

#         # --- Load Pre-Trained ML Model (or a dummy placeholder) ---
#         ml_model_weights_path = os.path.join(getattr(settings, 'MODEL_PATH_ROOT', ''), "eye_emotion_lstm_model.pt")
#         if not os.path.exists(ml_model_weights_path):
#             logging.warning(f"ML model weights not found at {ml_model_weights_path}. ML inference will be skipped.")
#             self.ml_model_loaded = False
#             self.RUN_HYBRID_MODEL_LOGIC = False # Disable hybrid if ML model isn't available
#         else:
#             try:
#                 self.ml_model_instance = EyeEmotionLSTM(
#                     input_dim=ML_INPUT_DIM, # or load from a config if selected features are used
#                     hidden_dim=ML_HIDDEN_DIM_LSTM,
#                     num_layers=ML_NUM_LAYERS_LSTM,
#                     num_classes=NUM_ML_EMOTION_CLASSES_V2_5,
#                     dropout_rate=ML_DROPOUT_RATE_MODEL
#                 ).to(self.device)
#                 self.ml_model_instance.load_state_dict(torch.load(ml_model_weights_path, map_location=self.device))
#                 self.ml_model_instance.eval()
#                 self.ml_model_loaded = True
#                 logging.info("PyTorch EyeEmotionLSTM model loaded successfully.")

#                 # In a real app, selected_ml_features_final and optimized_ml_hps_final
#                 # would also be loaded from saved config files here.
#                 # For this example, if no model is loaded, we implicitly use all features (ML_INPUT_DIM).
#                 # If a model is loaded, assume it expects ML_INPUT_DIM features, or load selected_features config.
#                 # self.ml_model_selected_features = [0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15] # Example if pre-selected
#                 # self.optimized_ml_hps = {'hidden_dim': 128, 'learning_rate': 0.001} # Example if pre-tuned

#             except Exception as e:
#                 logging.critical(f"Error loading PyTorch ML model: {e}")
#                 self.ml_model_loaded = False
#                 self.ml_model_instance = None
#                 self.RUN_HYBRID_MODEL_LOGIC = False

#         # --- Load Pre-Optimized Rule Thresholds (or use defaults) ---
#         # In a real app, this would load from a JSON/YAML file after an offline NIA run.
#         # For this example, we'll just use DEFAULTs or a hardcoded example if NIA isn't run offline.
#         # self.optimized_rules = {'EAR_THRESHOLD_BLINK': 0.25, ...} # Example of loaded optimized rules
#         if not self.optimized_rules: # If not loaded from file, use defaults
#             self.optimized_rules = {
#                 'EAR_THRESHOLD_BLINK': DEFAULT_EAR_THRESHOLD_BLINK,
#                 'EAR_CONSEC_FRAMES_BLINK': DEFAULT_EAR_CONSEC_FRAMES_BLINK,
#                 'EAR_THRESHOLD_DROWSY': DEFAULT_EAR_THRESHOLD_DROWSY,
#                 'FIXATION_MAX_MOVEMENT_PX': DEFAULT_FIXATION_MAX_MOVEMENT_PX,
#                 'T_BLINK_RATE_NERVOUS_HIGH_STATE': DEFAULT_T_BLINK_RATE_NERVOUS_HIGH_STATE,
#                 'FIXATION_STABILITY_MAX_VARIANCE': DEFAULT_FIXATION_STABILITY_MAX_VARIANCE,
#                 'T_FIX_STABILITY_GOOD_STATE_FACTOR': DEFAULT_T_FIX_STABILITY_GOOD_STATE_FACTOR,
#                 'T_SACCADE_AMP_SCANNING_HIGH_STATE': DEFAULT_T_SACCADE_AMP_SCANNING_HIGH_STATE
#             }
#         logging.info("Rule-based thresholds loaded (defaults or pre-optimized).")

#     def _process_video_logic(self, input_video_path: str, output_video_path: str,
#                              detector_instance, predictor_instance, # Dlib instances passed here
#                              active_rule_thresholds: Optional[Dict[str, Any]],
#                              ml_model_instance: Optional[EyeEmotionLSTM],
#                              ml_model_selected_features: Optional[List[int]],
#                              run_hybrid_logic: bool,
#                              apply_rule_smoothing: bool,
#                              rule_smoothing_window: int,
#                              frame_skip_config: int = FRAME_SKIP,
#                              max_processing_fps_config: float = MAX_PROCESSING_FPS
#                              ) -> Dict:
#         logging.info(f"{SCRIPT_VERSION} Processing: '{os.path.basename(input_video_path)}' -> '{os.path.basename(output_video_path)}'")

#         cap = cv2.VideoCapture(input_video_path)
#         if not cap.isOpened(): raise RuntimeError(f"Can't open video: {input_video_path}")
#         o_fps,fw,fh,tot_in_fr = cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         v_fps = o_fps if o_fps>0 else DEFAULT_PROCESSING_FPS
#         t_fps = min(v_fps, float(max_processing_fps_config))
#         a_skip = max(1, int(round(v_fps/t_fps))) if t_fps>0 else max(1, int(frame_skip_config))
#         eff_fps = (v_fps/a_skip) if a_skip>0 else DEFAULT_PROCESSING_FPS
#         logging.info(f"Vid: {fw}x{fh}@{o_fps:.1f}FPS ({tot_in_fr}fr). Proc @~{eff_fps:.1f}FPS (Skip:{a_skip})")

#         temp_output_path = output_video_path.replace('.mp4', '_temp.mp4')
#         try:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out_w = cv2.VideoWriter(temp_output_path, fourcc, eff_fps, (fw,fh))
#             if not out_w.isOpened(): raise RuntimeError(f"Can't create writer for {temp_output_path}. Codec: mp4v")
#         except Exception as e:
#             cap.release(); raise RuntimeError(f"VideoWriter init error: {e}")

#         feat_ext = EyeFeatureExtractor(fps=eff_fps)
#         state_mgr = EyeStateManager(fps=eff_fps, initial_thresholds=active_rule_thresholds)
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if ml_model_instance: ml_model_instance.to(device).eval()

#         fr_idx, p_count, s_t, ll_t = 0,0,time.time(),time.time()
#         all_features_session: List[Dict[str, Any]] = []
#         primary_detailed_state_counts: Counter[str] = Counter()
#         primary_emotion_counts: Counter[str] = Counter()

#         rule_state_hist: Deque[str] = deque(maxlen=max(1, rule_smoothing_window if apply_rule_smoothing else 1))
#         ml_feature_seq_hist: Deque[np.ndarray] = deque(maxlen=ML_SEQ_LENGTH)

#         while cap.isOpened():
#             ret,frame=cap.read()
#             if not ret: break
#             if fr_idx % a_skip != 0: fr_idx+=1; continue
#             ts=fr_idx/v_fps if v_fps>0 else p_count/eff_fps
#             gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#             faces=detector_instance(gray,0) # Use passed instance

#             current_features: Dict[str,Any] = {}
#             face_rect_draw: Optional[Tuple[int,int,int,int]] = None
#             landmarks_draw = None

#             raw_rule_state, rule_confidence = "No Face", 0.95
#             smoothed_rule_state = raw_rule_state

#             ml_emotion_pred, ml_confidence = "N/A", 0.0
#             ml_pred_probs_np: Optional[np.ndarray] = None

#             final_emotion_to_display = map_detailed_state_to_emotion(smoothed_rule_state)
#             final_state_to_display = smoothed_rule_state
#             final_confidence_to_display = rule_confidence
#             final_source = "Rule"

#             raw_rule_info_str: Optional[str] = None
#             raw_ml_info_str: Optional[str] = None

#             if faces:
#                 face = faces[0]
#                 face_rect_draw = (face.left(), face.top(), face.right(), face.bottom())
#                 landmarks_draw = predictor_instance(gray, face) # Use passed instance
#                 feat_ext.update(landmarks_draw, face, ts, frame.shape, state_mgr.thresholds)
#                 current_features = feat_ext.get_features_dict()
#                 all_features_session.append(current_features.copy())

#                 raw_rule_state, rule_confidence = state_mgr.classify_with_confidence(current_features)
#                 if apply_rule_smoothing:
#                     rule_state_hist.append(raw_rule_state)
#                     smoothed_rule_state = Counter(rule_state_hist).most_common(1)[0][0]
#                 else:
#                     smoothed_rule_state = raw_rule_state
#                 raw_rule_info_str = f"{raw_rule_state[:10]} ({rule_confidence:.2f})"
#                 if apply_rule_smoothing and raw_rule_state != smoothed_rule_state:
#                     raw_rule_info_str += f" ->Sm:{smoothed_rule_state[:10]}"

#                 if ml_model_instance:
#                     numerical_features = feat_ext.get_numerical_feature_vector()
#                     if ml_model_selected_features:
#                         if len(numerical_features) >= max(ml_model_selected_features) + 1 :
#                             numerical_features = numerical_features[ml_model_selected_features]
#                         else:
#                             logging.warning("ML feature selection indices out of bounds for current features. Using all features for ML prediction.")

#                     ml_feature_seq_hist.append(numerical_features)
#                     if len(ml_feature_seq_hist) == ML_SEQ_LENGTH:
#                         seq_tensor = torch.tensor(np.array(list(ml_feature_seq_hist)), dtype=torch.float32).unsqueeze(0).to(self.device) # use self.device
#                         with torch.no_grad():
#                             outputs = ml_model_instance(seq_tensor)
#                             ml_pred_probs = softmax(outputs.cpu().numpy().flatten())
#                             ml_pred_probs_np = ml_pred_probs
#                             top_idx = np.argmax(ml_pred_probs)
#                             ml_emotion_pred = EMOTION_CLASSES_V2_5[top_idx]
#                             ml_confidence = ml_pred_probs[top_idx]
#                         raw_ml_info_str = f"{ml_emotion_pred[:10]} ({ml_confidence:.2f})"

#                 if run_hybrid_logic and ml_model_instance and ml_pred_probs_np is not None:
#                     final_emotion_to_display, final_confidence_to_display, final_source = hybrid_fusion_strategy(
#                         smoothed_rule_state, rule_confidence,
#                         ml_pred_probs_np, lambda idx: EMOTION_CLASSES_V2_5[idx]
#                     )
#                     final_state_to_display = final_emotion_to_display
#                 else:
#                     final_emotion_to_display = map_detailed_state_to_emotion(smoothed_rule_state)
#                     final_state_to_display = smoothed_rule_state
#                     final_confidence_to_display = rule_confidence
#                     final_source = "Rule(S)" if apply_rule_smoothing else "Rule"
#             else: # No face detected
#                 feat_ext.reset_state()
#                 current_features = feat_ext.get_features_dict()
#                 raw_rule_state, rule_confidence = state_mgr.classify_with_confidence(current_features)
#                 if apply_rule_smoothing:
#                     rule_state_hist.append(raw_rule_state)
#                     smoothed_rule_state = Counter(rule_state_hist).most_common(1)[0][0]
#                 else: smoothed_rule_state = raw_rule_state

#                 final_emotion_to_display = map_detailed_state_to_emotion(smoothed_rule_state)
#                 final_state_to_display = smoothed_rule_state
#                 final_confidence_to_display = rule_confidence
#                 final_source = "Rule(S)" if apply_rule_smoothing else "Rule"
#                 ml_feature_seq_hist.clear()

#             primary_detailed_state_counts[final_state_to_display] +=1
#             primary_emotion_counts[final_emotion_to_display] +=1

#             overall_disp_text = "Analyzing..." if p_count < (eff_fps * 5) else None
#             draw_diagnostics_on_frame_v2_5(frame, current_features,
#                                             final_emotion_to_display, final_state_to_display, final_confidence_to_display, final_source,
#                                             face_rect_draw, landmarks_draw, overall_disp_text,
#                                             raw_rule_info=raw_rule_info_str, raw_ml_info=raw_ml_info_str)
#             out_w.write(frame); p_count+=1; fr_idx+=1
#             if time.time()-ll_t >=10.0:
#                 el = time.time()-s_t; cur_fps = p_count/el if el>0 else float('inf')
#                 prog = (p_count / ((tot_in_fr/a_skip) if tot_in_fr>0 and a_skip>0 else p_count or 1)) * 100
#                 logging.info(f"Proc: {p_count} fr ({prog:.1f}%). FPS: {cur_fps:.1f}. HybridState: {final_state_to_display[:15]}")
#                 ll_t=time.time()

#         cap.release(); out_w.release(); cv2.destroyAllWindows()
#         tot_t=time.time()-s_t; logging.info(f"Vid proc fin: {tot_t:.1f}s for {p_count} frames.")
#         if p_count==0: logging.warning("No frames processed."); return {}

#         try:
#             ffmpeg_cmd = [
#                 'ffmpeg', '-y', '-i', temp_output_path, '-c:v', 'libx264', '-preset', 'fast',
#                 '-movflags', '+faststart', '-pix_fmt', 'yuv420p', output_video_path
#             ]
#             subprocess.run(ffmpeg_cmd, check=True)
#             #os.remove(temp_output_path)  # Remove temp file after successful conversion
#         except Exception as e:
#             logging.error(f"FFmpeg conversion failed: {e}", exc_info=True)
#             try: os.remove(temp_output_path)  # Clean up temp file even if conversion fails
#             except Exception as clean_e: logging.warning(f"Failed to delete temp output video after FFmpeg error: {clean_e}")
#             raise RuntimeError(f"FFmpeg conversion failed: {e}")
        

#         report_content_for_pdf = {}
#         overall_emo, overall_expl = state_mgr.determine_overall_condition(primary_detailed_state_counts,p_count)
#         report_content_for_pdf["overall_summary"] = {"emotion": overall_emo, "explanation": overall_expl}
#         report_content_for_pdf["emotion_distribution"] = primary_emotion_counts

#         avg_f_dict:Dict[str,Any]={};
#         if all_features_session and all_features_session[0]:
#             tmp_agg:Dict[str,List] = {f'avg_{k}':[] for k in all_features_session[0] if isinstance(all_features_session[0][k],(int,float))}
#             for f_dict in all_features_session:
#                 for k,v in f_dict.items():
#                     if isinstance(v,(int,float)) and f'avg_{k}' in tmp_agg : tmp_agg[f'avg_{k}'].append(v)
#             for k,vals in tmp_agg.items(): avg_f_dict[k]=np.mean(vals) if vals else None
#         report_content_for_pdf["average_features"] = avg_f_dict

#         report_content_for_pdf["processing_summary"] = {
#             "processed_frames":p_count,"total_input_frames":tot_in_fr if tot_in_fr>0 else "Unk",
#             "avg_processing_fps":p_count/tot_t if tot_t>0 else 0,
#             "smoothing_window": f"{rule_smoothing_window} frames" if apply_rule_smoothing else "Disabled"
#         }

#         print("\n"+"="*70 + f"\n EYE BEHAVIOR ANALYSIS SUMMARY ({SCRIPT_VERSION})\n"+"="*70)
#         print(f" Input Video: {os.path.basename(input_video_path)}")
#         print(f" Output Video: {os.path.basename(output_video_path)}")
#         print(f" Frames Processed: {report_content_for_pdf['processing_summary']['processed_frames']}")
#         print(f" Overall Emotion (Primary System): {report_content_for_pdf['overall_summary']['emotion']}")
#         print(f" Explanation: {report_content_for_pdf['overall_summary']['explanation']}\n" + "-"*70)
#         print(" Top 3 Emotions (Primary System - % of processed frames):")
#         for i,(emo,ct) in enumerate(report_content_for_pdf.get("emotion_distribution", Counter()).most_common(3)):
#             print(f"  {i+1}. {emo}: {ct} frames ({(ct/p_count)*100:.1f}%)")
#         print("="*70 + "\n")
#         logging.info("--- Processing & Reporting Complete ---")

    

#         return report_content_for_pdf

#     def post(self, request):
#         logging.info(f"--- Advanced Eye Model {SCRIPT_VERSION} Initiating Analysis from Django View ---")

#         if not self.dlib_loaded:
#             return JsonResponse({"status": "error", "message": "Dlib models failed to load at server startup. Cannot process video."}, status=500)
#         if self.RUN_HYBRID_MODEL_LOGIC and not self.ml_model_loaded:
#             return JsonResponse({"status": "error", "message": "ML model failed to load at server startup, and hybrid logic is enabled. Cannot proceed."}, status=500)


#         video_path = request.session.get('uploaded_video_path')
#         if not video_path:
#             return JsonResponse({"error": "No video file found in session"}, status=400)

#         video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
#         if not os.path.exists(video_absolute_path):
#             return JsonResponse({"error": "Video file not found on server"}, status=404)

#         # Get basename and extension for output files
#         input_file_basename = os.path.splitext(os.path.basename(video_path))[0] 
#         input_file_ext = os.path.splitext(video_path)[1]
        
#         # Setup input path variables
#         input_vid_full_path = video_absolute_path
#         input_vid_path_relative = video_path

#         logging.info(f"Processing video from: {input_vid_full_path}")

#         video_report_data = {}
#         output_video_full_path = ""
#         output_pdf_full_path = ""

#         if self.RUN_VIDEO_PROCESSING:
#             output_vid_suffix_parts = ["analyzed", SCRIPT_VERSION.lower().replace(' ', '_')]
#             output_vid_suffix_parts.append("rules_opt" if self.optimized_rules else "rules_def") # Use optimized rules if available
#             if self.APPLY_RULE_TEMPORAL_SMOOTHING: output_vid_suffix_parts.append("smooth")
#             if self.RUN_HYBRID_MODEL_LOGIC and self.ml_model_loaded: output_vid_suffix_parts.append("hybrid")
#             else: output_vid_suffix_parts.append("rules_only") # Indicate if ML is off or failed

#             output_video_filename = f"{input_file_basename}_{'_'.join(output_vid_suffix_parts)}{input_file_ext}"
#             processed_videos_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
#             os.makedirs(processed_videos_dir, exist_ok=True)
#             output_video_full_path = os.path.join(processed_videos_dir, output_video_filename)

#             try:
#                 video_report_data = self._process_video_logic(
#                     input_video_path=input_vid_full_path,
#                     output_video_path=output_video_full_path,
#                     detector_instance=self.detector, # Pass pre-loaded instances
#                     predictor_instance=self.predictor,
#                     active_rule_thresholds=self.optimized_rules,
#                     ml_model_instance=self.ml_model_instance if self.RUN_HYBRID_MODEL_LOGIC else None,
#                     ml_model_selected_features=self.ml_model_selected_features if self.RUN_HYBRID_MODEL_LOGIC else None,
#                     run_hybrid_logic=self.RUN_HYBRID_MODEL_LOGIC,
#                     apply_rule_smoothing=self.APPLY_RULE_TEMPORAL_SMOOTHING,
#                     rule_smoothing_window=RULE_BASED_SMOOTHING_WINDOW_SIZE
#                 )

#                 overall_emotion = video_report_data.get("overall_summary", {}).get("emotion", "N/A")
#             except Exception as e:
                
#                 return JsonResponse({"status": "error", "message": f"Video processing failed: {e}"}, status=500)
#         else:
#             logging.info("Video processing was skipped based on configuration.")
#             video_report_data["processing_summary"] = {"processed_frames": 0, "total_input_frames": "N/A", "avg_processing_fps": 0}
#             video_report_data["overall_summary"] = {"emotion": "N/A", "explanation": "Video processing not run."}
#             video_report_data["emotion_distribution"] = Counter()
#             video_report_data["average_features"] = {}

#         if self.RUN_PRELIMINARY_VALIDATION:
#             logging.info(f"--- {SCRIPT_VERSION}: Running Preliminary Validations ---")
#             dummy_test_data_rb: List[Dict[str, Any]] = []
#             for i in range(ML_DUMMY_NUM_TEST_SAMPLES):
#                 gt_emo = random.choice(EMOTION_CLASSES_V2_5)
#                 dummy_feats = {'avg_ear': random.uniform(0.1,0.4), 'blink_rate_bpm': random.uniform(3,35),
#                                'current_fixation_frames': float(random.randint(0, int(DEFAULT_PROCESSING_FPS*2))),
#                                'fixation_stability': random.uniform(5,100) if random.random() > 0.1 else -1.0,
#                                'avg_saccade_amplitude': random.uniform(1,30),
#                                'inter_ocular_dist_face_ratio': random.uniform(0.1,0.2),
#                                'pupil_proxy_derivative1': random.uniform(-0.05, 0.05) * DEFAULT_PROCESSING_FPS,
#                                'head_roll': random.uniform(-15,15), 'head_yaw': random.uniform(-15,15),
#                                'is_blinking_now':0.0, 'is_drowsy_closure':0.0}
#                 dummy_test_data_rb.append({"id": f"rb_test_{i}", "features": dummy_feats, "ground_truth_emotion": gt_emo})

#             temp_state_mgr_for_val = EyeStateManager(fps=DEFAULT_PROCESSING_FPS, initial_thresholds=self.optimized_rules)
#             rb_val_results = validate_rule_based_system(temp_state_mgr_for_val, dummy_test_data_rb, "Dummy Test Set RB")
#             video_report_data["validation_report_rule_based"] = rb_val_results

#             if self.ml_model_loaded:
#                 ml_test_seqs, ml_test_labels, _ = generate_dummy_ml_data(ML_DUMMY_NUM_TEST_SAMPLES, ML_SEQ_LENGTH, ML_INPUT_DIM, NUM_ML_EMOTION_CLASSES_V2_5, is_test_set=True)
#                 input_dim_for_ml_val = ML_INPUT_DIM
#                 if self.ml_model_selected_features:
#                     ml_test_seqs_sel = [s[:, self.ml_model_selected_features] for s in ml_test_seqs if s.ndim==2 and s.shape[1]>=len(self.ml_model_selected_features)]
#                     if ml_test_seqs_sel: ml_test_seqs = ml_test_seqs_sel
#                     input_dim_for_ml_val = len(self.ml_model_selected_features)

#                 if input_dim_for_ml_val > 0:
#                     ml_val_results = validate_ml_model(self.ml_model_instance, ml_test_seqs, ml_test_labels,
#                                                        "Dummy Test Set ML", self.device,
#                                                        input_dim_for_model=input_dim_for_ml_val)
#                     video_report_data["validation_report_ml"] = ml_val_results
#                 else:
#                     video_report_data["validation_report_ml"] = {"source": "Dummy Test Set ML", "num_samples": 0, "report_str": "ML validation skipped due to 0 input features."}
#             else:
#                 logging.info("ML model not available for validation.")
#                 video_report_data["validation_report_ml"] = {"source": "Dummy Test Set ML", "num_samples": 0, "report_str": "ML model not trained/loaded."}

#         pdf_report_name_base = os.path.splitext(os.path.basename(input_vid_full_path))[0]
#         reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
#         os.makedirs(reports_dir, exist_ok=True)
#         final_pdf_output_path = os.path.join(reports_dir, f"{slugify(pdf_report_name_base)}{PDF_REPORT_FILENAME_SUFFIX}")

#         try:
#           generate_pdf_report_reportlab_v2_5(final_pdf_output_path, os.path.basename(input_vid_full_path), video_report_data)
#           output_pdf_full_path = final_pdf_output_path
#         except Exception as e:
#             logging.error(f"PDF report generation failed: {e}", exc_info=True)
#             try: default_storage.delete(input_vid_path_relative)
#             except Exception as clean_e: logging.warning(f"Failed to delete temp input video after PDF error: {clean_e}")
#             return JsonResponse({"status": "error", "message": f"PDF report generation failed: {e}"}, status=500)

        

#         result_data = {
#             "status": "success",
#             "message": "Eye analysis complete.",
#             "output_video_url": settings.MEDIA_URL + os.path.relpath(output_video_full_path, settings.MEDIA_ROOT) if self.RUN_VIDEO_PROCESSING and output_video_full_path else None,
#             "pdf_report_url": settings.MEDIA_URL + os.path.relpath(output_pdf_full_path, settings.MEDIA_ROOT) if output_pdf_full_path else None,
#             "overall_summary": video_report_data.get("overall_summary"),
#             "emotion_distribution": dict(video_report_data.get("emotion_distribution", Counter())),
#             "processing_summary": video_report_data.get("processing_summary"),
#             "final_emotion": overall_emotion
#         }
#         return JsonResponse(result_data)
'''
# ------------------ eye_analysis v2 class ------------------
"""class  EyeAnalysisViewV2(View):
    def __init__(self, calibration_seconds=5, cool_down_period=2.0):
        # This class also remains unchanged.
        self.calibration_seconds = calibration_seconds
        self.cool_down_period = cool_down_period
        self.is_calibrated = False
        self.baseline_state = "Unknown"
        self.previous_metrics = {}
        self.current_emotion = "Calibrating"
        self.current_confidence = 0
        self.emotion_cool_down_until = 0
        self._calibration_data = []
        self.dat_file_path = "/home/student/new_api/testpro-main/models/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.dat_file_path)

    def encode_video(self,input_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding failed: {process.stderr.decode()}")
                return None

            return output_path

        except Exception as e:
            print(f"Error during video encoding: {e}")
            return None
        

    def _finalize_calibration(self):
        if self._calibration_data:
            self.baseline_state = Counter(state['eye_state'] for state in self._calibration_data).most_common(1)[0][0]
        else:
            self.baseline_state = "Natural"
        self.is_calibrated = True
        self.current_emotion = "Calm"
        print(f"-> Calibration complete. Baseline eye state established as: {self.baseline_state}")

    def _calculate_confidence(self, metrics, head_stability):
        confidence = 65
        if head_stability == 'Stable': confidence += 15
        if metrics['gaze_direction'] == 'Center': confidence += 10
        if metrics['eye_state'] in ['Intense', 'Nervous']: confidence += 5
        return min(confidence, 99)
    
    def update(self, metrics, head_stability, frame_time):
        if not metrics:
            return self.current_emotion, self.current_confidence
        if not self.is_calibrated:
            self._calibration_data.append(metrics)
            if frame_time >= self.calibration_seconds:
                self._finalize_calibration()
            return "Calibrating...", 0
        if frame_time < self.emotion_cool_down_until:
            return self.current_emotion, self.current_confidence
        new_emotion = self.current_emotion
        prev_state = self.previous_metrics.get('eye_state', self.baseline_state)
        if metrics['eye_state'] != prev_state:
            if metrics['eye_state'] == "Nervous" and new_emotion != "Agitated": new_emotion = "Agitated"
            elif metrics['eye_state'] in ["Focused", "Intense"] and prev_state in ["Natural", "Distracted"]: new_emotion = "Engaged"
            elif metrics['eye_state'] in ["Natural", "Distracted"] and prev_state in ["Focused", "Intense"]: new_emotion = "Disengaged"
            elif metrics['eye_state'] == self.baseline_state and new_emotion != "Calm": new_emotion = "Calm"
        if new_emotion != self.current_emotion:
            self.current_emotion = new_emotion
            self.current_confidence = self._calculate_confidence(metrics, head_stability)
            self.emotion_cool_down_until = frame_time + self.cool_down_period
        self.previous_metrics = metrics
        return self.current_emotion, self.current_confidence

    def analyze_facial_metrics(self,landmarks, frame, previous_eye_center, fixation_duration):
        # This function remains the same as it's the core of the analysis.
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        interpupillary_distance = np.linalg.norm(left_eye.mean(axis=0) - right_eye.mean(axis=0))
        face_width = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
        ipd_ratio = interpupillary_distance / face_width if face_width > 0 else 0.4
        current_eye_center = (left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2
        if previous_eye_center is not None:
            movement = np.linalg.norm(current_eye_center - previous_eye_center)
            fixation_duration = fixation_duration + 1 if movement < 5 else 0
        else:
            fixation_duration = 0
        previous_eye_center = current_eye_center
        if ipd_ratio > 0.42 and fixation_duration > 20: eye_state = "Intense"
        elif ipd_ratio > 0.42 and fixation_duration < 10: eye_state = "Focused"
        elif ipd_ratio < 0.38 and fixation_duration > 20: eye_state = "Nervous"
        elif ipd_ratio < 0.38 and fixation_duration < 10: eye_state = "Distracted"
        else: eye_state = "Natural"
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        face_center_x = (landmarks.part(16).x + landmarks.part(0).x) / 2
        gaze_ratio = (nose_tip[0] - face_center_x) / face_width if face_width > 0 else 0
        if gaze_ratio < -0.05: gaze_direction = "Right"
        elif gaze_ratio > 0.05: gaze_direction = "Left"
        else: gaze_direction = "Center"
        for i in range(36, 48):
            p = landmarks.part(i)
            cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
        return {
            "eye_state": eye_state, "ipd_ratio": ipd_ratio,
            "gaze_direction": gaze_direction, "fixation_duration": fixation_duration
        }, previous_eye_center

    def process_video(self,input_video, output_video, mtcnn, predictor):
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        previous_eye_center, fixation_duration, previous_face_box_center = None, 0, None
        emotion_counts, eye_state_counts = Counter(), Counter()
        
        start_time = time.time()
        
        with Progress(
            TextColumn("[blue]Processing:"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(), console=console
        ) as progress:
            task = progress.add_task("Frames", total=total_frames)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)
                facial_metrics, head_stability = {}, "Unknown"

                if boxes is not None:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    current_face_box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    if previous_face_box_center is not None:
                        head_movement = np.linalg.norm(current_face_box_center - previous_face_box_center)
                        head_stability = "Stable" if head_movement < 10 else "Unstable"
                    else: head_stability = "Stable"
                    previous_face_box_center = current_face_box_center
                    landmarks = predictor(frame_rgb, dlib.rectangle(x1, y1, x2, y2))
                    facial_metrics, previous_eye_center = self.analyze_facial_metrics(
                        landmarks, frame, previous_eye_center, fixation_duration)
                    fixation_duration = facial_metrics['fixation_duration']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                video_time = time.time() - start_time
                probable_emotion, confidence = self.update(facial_metrics, head_stability, video_time)
                
                if self.is_calibrated and facial_metrics:
                    emotion_counts[probable_emotion] += 1
                    eye_state_counts[facial_metrics['eye_state']] += 1

                font_scale, font_thickness = 0.5, 1
                cv2.putText(frame, f"Emotion: {probable_emotion} (Conf: {confidence:.0f}%)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.1, (0, 0, 255), font_thickness + 1)
                
                out.write(frame)
                progress.update(task, advance=1)
        
        cap.release(), out.release()
        
        overall_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Not Determined"
        overall_eye_state = eye_state_counts.most_common(1)[0][0] if eye_state_counts else "Not Determined"

        return {
            "overall_emotion": overall_emotion,
            "overall_eye_state": overall_eye_state,
            "baseline_state": self.baseline_state
        }

    def post(self, request):
        logging.info(f"--- Eye Analysis V2 Initiating Analysis from Django View ---")

        if not self.dat_file_path:
            return JsonResponse({"status": "error", "message": "Dlib models failed to load at server startup. Cannot process video."}, status=500)

        video_path = request.session.get('uploaded_video_path')
        if not video_path:
            return JsonResponse({"error": "No video file found in session"}, status=400)

        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_absolute_path):
            return JsonResponse({"error": "Video file not found on server"}, status=404)

        video_uuid = request.session.get('video_uuid')
        if not video_uuid:
            return JsonResponse({"error": "No video UUID found in session"}, status=400)
        
        input_vid_full_path = video_absolute_path

        logging.info(f"Processing video from: {input_vid_full_path}")

        output_temp = "eye_temp.mp4"

        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_temp)


        output_video_full_path = output_path
        try:
            report_data = self.process_video(
                input_video=input_vid_full_path,
                output_video=output_video_full_path,
                mtcnn = MTCNN(keep_all=False, device=device),
                predictor=self.predictor
            )
            overall_emotion = report_data.get("overall_emotion", "N/A")
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Video processing failed: {e}"}, status=500)

        output_filename = f"eye_analysis_video_{video_uuid}.mp4"
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_final_path = os.path.join(output_dir, output_filename)

        self.encode_video(output_video_full_path,output_final_path)

        result_data = {
            "status": "success",
            "message": "Eye analysis complete.",
            "output_video_url": settings.MEDIA_URL + os.path.relpath(output_final_path, settings.MEDIA_ROOT) if output_final_path else None,
            "overall_emotion": overall_emotion,
            "overall_eye_state": report_data.get("overall_eye_state", "N/A"),
            "baseline_state": report_data.get("baseline_state", "N/A")
        }
        return JsonResponse(result_data)
"""

# ------------------ eye_analysis v3 class ------------------

class PsychologicalEmotionModel:
    # This class remains unchanged.
    def __init__(self, calibration_seconds=5, cool_down_period=2.0):
        self.calibration_seconds = calibration_seconds
        self.cool_down_period = cool_down_period
        self.is_calibrated = False
        self.baseline_state = "Unknown"
        self.previous_metrics = {}
        self.current_emotion = "Calibrating"
        self.current_confidence = 0
        self.emotion_cool_down_until = 0
        self._calibration_data = []

    def _finalize_calibration(self):
        if self._calibration_data:
            self.baseline_state = Counter(state['eye_state'] for state in self._calibration_data).most_common(1)[0][0]
        else: self.baseline_state = "Natural"
        self.is_calibrated = True
        self.current_emotion = "Calm"
        print(f"-> Calibration complete. Baseline eye state established as: {self.baseline_state}")

    def _calculate_confidence(self, metrics, head_stability):
        confidence = 65
        if head_stability == 'Stable': confidence += 15
        if metrics.get('gaze_direction') == 'Center': confidence += 10
        if metrics.get('eye_state') in ['Intense', 'Nervous']: confidence += 5
        return min(confidence, 99)

    def update(self, metrics, head_stability, frame_time):
        if not metrics: return self.current_emotion, self.current_confidence
        if not self.is_calibrated:
            self._calibration_data.append(metrics)
            if frame_time >= self.calibration_seconds: self._finalize_calibration()
            return "Calibrating...", 0
        if frame_time < self.emotion_cool_down_until: return self.current_emotion, self.current_confidence
        new_emotion = self.current_emotion
        prev_state = self.previous_metrics.get('eye_state', self.baseline_state)
        if metrics['eye_state'] != prev_state:
            if metrics['eye_state'] == "Nervous" and new_emotion != "Agitated": new_emotion = "Agitated"
            elif metrics['eye_state'] in ["Focused", "Intense"] and prev_state in ["Natural", "Distracted"]: new_emotion = "Engaged"
            elif metrics['eye_state'] in ["Natural", "Distracted"] and prev_state in ["Focused", "Intense"]: new_emotion = "Disengaged"
            elif metrics['eye_state'] == self.baseline_state and new_emotion != "Calm": new_emotion = "Calm"
        if new_emotion != self.current_emotion:
            self.current_emotion = new_emotion
            self.current_confidence = self._calculate_confidence(metrics, head_stability)
            self.emotion_cool_down_until = frame_time + self.cool_down_period
        self.previous_metrics = metrics
        return self.current_emotion, self.current_confidence

class EyeAnalysisViewV3(View):
    def encode_video(self,input_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding failed: {process.stderr.decode()}")
                return None

            return output_path

        except Exception as e:
            print(f"Error during video encoding: {e}")
            return None
        
    def analyze_facial_metrics(self,landmarks, frame, previous_eye_center, fixation_duration):
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        interpupillary_distance = np.linalg.norm(left_eye.mean(axis=0) - right_eye.mean(axis=0))
        face_width = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
        ipd_ratio = interpupillary_distance / face_width if face_width > 0 else 0.4
        current_eye_center = (left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2
        if previous_eye_center is not None:
            movement = np.linalg.norm(current_eye_center - previous_eye_center)
            fixation_duration = fixation_duration + 1 if movement < 5 else 0
        else:
            fixation_duration = 0
        previous_eye_center = current_eye_center
        if ipd_ratio > 0.42 and fixation_duration > 20: eye_state = "Intense"
        elif ipd_ratio > 0.42 and fixation_duration < 10: eye_state = "Focused"
        elif ipd_ratio < 0.38 and fixation_duration > 20: eye_state = "Nervous"
        elif ipd_ratio < 0.38 and fixation_duration < 10: eye_state = "Distracted"
        else: eye_state = "Natural"
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        face_center_x = (landmarks.part(16).x + landmarks.part(0).x) / 2
        gaze_ratio = (nose_tip[0] - face_center_x) / face_width if face_width > 0 else 0
        if gaze_ratio < -0.05: gaze_direction = "Right"
        elif gaze_ratio > 0.05: gaze_direction = "Left"
        else: gaze_direction = "Center"
        for i in range(36, 48):
            p = landmarks.part(i)
            cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
        return {
            "eye_state": eye_state, "ipd_ratio": ipd_ratio,
            "gaze_direction": gaze_direction, "fixation_duration": fixation_duration
        }, previous_eye_center
    

    # --- OTHER PARAMETER CALCULATION (UNCHANGED) ---
    def calculate_other_parameters(self,landmarks, frame_shape):
        def get_ear(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.0

        left_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        ear = (get_ear(left_eye_pts) + get_ear(right_eye_pts)) / 2.0
        blink_status = "Blink" if ear < 0.2 else "Open"

        mouth_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(60, 68)])
        A = np.linalg.norm(mouth_pts[2] - mouth_pts[6])
        B = np.linalg.norm(mouth_pts[0] - mouth_pts[4])
        mar = A / B if B > 0 else 0

        image_points = np.array([(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x, landmarks.part(8).y), (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))
        (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        return {"ear": ear, "blink_status": blink_status, "mar": mar, "head_pose_p1": p1, "head_pose_p2": p2}

    # --- REAL-TIME DISPLAY FUNCTION (MODIFIED) ---
    def draw_analysis_panel(self,frame, data):
        panel_width = 280  # Adjusted width for smaller font
        panel_height = frame.shape[0]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # --- MODIFIED: Smaller font and tighter line spacing ---
        y_pos = 25
        line_height = 18 
        font_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        def draw_text(text, value, unit="", color=(255, 255, 255)):
            nonlocal y_pos
            display_text = f"{text}: {value} {unit}"
            cv2.putText(frame, display_text, (10, y_pos), font, font_scale, color, 1, cv2.LINE_AA)
            y_pos += line_height

        def draw_header(text):
            nonlocal y_pos
            y_pos += 8
            cv2.putText(frame, text, (10, y_pos), font, font_scale + 0.1, (100, 255, 255), 1, cv2.LINE_AA)
            y_pos += (line_height + 4)

        # --- MODIFIED: Unified panel, no "rogue" vs "core" distinction ---
        draw_header("--- Final Determination ---")
        draw_text("Emotion", data.get("emotion", "N/A"), color=(0, 0, 255))
        draw_text("Confidence", f"{data.get('confidence', 0):.0f}%", color=(0, 255, 0))
        
        draw_header("--- Analysis Metrics ---")
        draw_text("Eye State", data.get("eye_state", "N/A"))
        draw_text("Blink Status", data.get("blink_status", "N/A"))
        draw_text("Eye Aspect Ratio", f"{data.get('ear', 0):.3f}")
        draw_text("IPD-to-Face Ratio", f"{data.get('ipd_ratio', 0):.3f}")
        draw_text("Gaze Direction", data.get('gaze_direction', "N/A"))
        draw_text("Fixation (frames)", f"{data.get('fixation_duration', 0)}")
        draw_text("Mouth Openness", f"{data.get('mar', 0):.3f}")
        draw_text("Head Stability", data.get('head_stability', "N/A"))
        draw_text("Head Pose", "See blue line")
        
        if data.get("head_pose_p1"):
            cv2.line(frame, data["head_pose_p1"], data["head_pose_p2"], (255, 100, 0), 2)
            
        return frame

    # --- VIDEO PROCESSING FUNCTION (MODIFIED) ---
    def process_video(self,input_video, output_video, mtcnn, predictor):
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        previous_eye_center, fixation_duration, previous_face_box_center = None, 0, None
        emotion_model = PsychologicalEmotionModel()
        emotion_counts, eye_state_counts = Counter(), Counter()
        
        start_time = time.time()
        
        with Progress(TextColumn("[blue]Processing:"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task("Frames", total=total_frames)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)
                
                facial_metrics, other_metrics, head_stability = {}, {}, "Unknown"
                
                if boxes is not None:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    current_face_box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    if previous_face_box_center is not None:
                        head_movement = np.linalg.norm(current_face_box_center - previous_face_box_center)
                        head_stability = "Stable" if head_movement < 10 else "Unstable"
                    else: head_stability = "Stable"
                    previous_face_box_center = current_face_box_center
                    
                    landmarks = predictor(frame_rgb, dlib.rectangle(x1, y1, x2, y2))
                    facial_metrics, previous_eye_center = self.analyze_facial_metrics(landmarks, frame, previous_eye_center, fixation_duration)
                    fixation_duration = facial_metrics['fixation_duration']
                    other_metrics = self.calculate_other_parameters(landmarks, frame.shape) # Renamed function call
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                video_time = time.time() - start_time
                probable_emotion, confidence = emotion_model.update(facial_metrics, head_stability, video_time)
                
                if emotion_model.is_calibrated and facial_metrics:
                    emotion_counts[probable_emotion] += 1
                    eye_state_counts[facial_metrics['eye_state']] += 1

                display_data = {"emotion": probable_emotion, "confidence": confidence, "head_stability": head_stability, **facial_metrics, **other_metrics}
                frame = self.draw_analysis_panel(frame, display_data)
                
                out.write(frame)
                progress.update(task, advance=1)
        
        cap.release(), out.release()
        
        overall_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Not Determined"
        overall_eye_state = eye_state_counts.most_common(1)[0][0] if eye_state_counts else "Not Determined"

        return {"overall_emotion": overall_emotion, "overall_eye_state": overall_eye_state, "baseline_state": emotion_model.baseline_state}

    # --- MAIN FUNCTION (UNCHANGED) ---
    def main_analysis(self,input_video_path,output_video_path):
        # --- Configuration: Hardcode your file paths here ---
        dat_file_path = "/home/student/analyze_video_audio/eye/shape_predictor_68_face_landmarks.dat"
        # ----------------------------------------------------

        print("\n--- Starting Eye Analysis ---")
        print("-> Initializing models...")
        if not os.path.exists(input_video_path):
            console.print(f"[bold red]Error: Input video not found at '{input_video_path}'[/bold red]")
            return
        if not os.path.exists(dat_file_path):
            console.print(f"[bold red]Error: Dlib's shape predictor file not found at '{dat_file_path}'[/bold red]")
            console.print("[bold yellow]Hint: Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place it in the correct directory.[/bold yellow]")
            return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=False, device=device)
        predictor = dlib.shape_predictor(dat_file_path)
        print(f"-> Models initialized on device: {device}")

        report_data = self.process_video(input_video_path, output_video_path, mtcnn, predictor)

        print("\n" + "-"*25)
        console.print("[bold green]Eye Analysis Complete[/bold green]")
        print("-"*25)
        console.print(f"[bold]Overall Subject Emotion:[/bold] [cyan]{report_data['overall_emotion']}[/cyan]")
        console.print(f"[bold]Overall Eye State:[/bold]       [yellow]{report_data['overall_eye_state']}[/yellow]")
        console.print(f"[bold]Baseline Eye State:[/bold]      {report_data['baseline_state']}")
        print("-"*25)
        console.print(f"-> Output video with real-time analysis saved to: [bold magenta]{output_video_path}[/bold magenta]")
        print("-" * 25 + "\n")

        return report_data['overall_emotion'],report_data['overall_eye_state'],report_data['baseline_state']

    def post(self,request):

        video_path = request.session.get('uploaded_video_path')
        if not video_path:
            return JsonResponse({"error": "No video file found in session"}, status=400)

        video_absolute_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_absolute_path):
            return JsonResponse({"error": "Video file not found on server"}, status=404)

        video_uuid = request.session.get('video_uuid')
        if not video_uuid:
            return JsonResponse({"error": "No video UUID found in session"}, status=400)
        
        input_vid_full_path = video_absolute_path


        output_temp = "eye_temp.mp4"

        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_temp)


        output_video_full_path = output_path

        overall_emotion,overall_eye_state,baseline_state = self.main_analysis(input_vid_full_path,output_video_full_path)

        output_filename = f"eye_analysis_video_{video_uuid}.mp4"
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_final_path = os.path.join(output_dir, output_filename)

        self.encode_video(output_video_full_path,output_final_path)

        result_data = {
            "status": "success",
            "message": "Eye analysis complete.",
            "output_video_url": settings.MEDIA_URL + os.path.relpath(output_final_path, settings.MEDIA_ROOT) if output_final_path else None,
            "overall_emotion": overall_emotion,
            "overall_eye_state": overall_eye_state,
            "baseline_state": baseline_state
        }
        return JsonResponse(result_data)






############################# eye_analysis class end ##############################

# ------------------ MergeVideos class ------------------
class MergeVideos(View):
    def __init__(self):
        super().__init__()

    def encode_video(self,input_path, output_path):
        try:
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print(f"FFmpeg encoding failed: {process.stderr.decode()}")
                return None

            return output_path

        except Exception as e:
            print(f"Error during video encoding: {e}")
            return None
        
    def add_audio(self,video_no_audio, audio_source, output_with_audio):
        command = [
            'ffmpeg',
            '-i', video_no_audio,
            '-i', audio_source,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            output_with_audio
        ]
        
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            print(f"[ERROR] FFmpeg stderr:\n{process.stderr.decode()}")
            raise Exception("Audio merging failed")
        return output_with_audio
    
    

    
    def merge_videos(self,video_paths, output_path):
        """
        Merges multiple videos by taking clips of calculated duration from the video to create an output video.

        Args:
            video_paths (list): List of paths to input video files.
            output_path (str): Path to save the output merged video.
        """
        random.shuffle(video_paths)  # Shuffle the video paths for randomness
        
        num_videos = len(video_paths)

        cap = cv2.VideoCapture(video_paths[0])
        if not cap.isOpened():
            raise ValueError("Failed to open video to compute duration.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps
        cap.release()
        clip_duration = total_duration / num_videos


        # Initialize video properties
        video_caps = []
        fps_list = []
        frame_counts = []
        widths, heights = [], []

        # Validate video files and collect properties
        for i, path in enumerate(video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Video file {path} is invalid or unsupported.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                raise ValueError(f"Video {path} has invalid FPS.")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                raise ValueError(f"Video {path} has no frames.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                cap.release()
                raise ValueError(f"Video {path} has invalid dimensions.")

            # Check if video has enough frames for its clip
            required_duration = (i + 1) * clip_duration
            if frame_count / fps < required_duration:
                cap.release()
                raise ValueError(f"Video {path} is too short. Needs at least {required_duration:.2f} seconds for clip {i * clip_duration:.2f}-{(i + 1) * clip_duration:.2f} seconds.")

            video_caps.append(cap)
            fps_list.append(fps)
            frame_counts.append(frame_count)
            widths.append(width)
            heights.append(height)

        # Ensure all videos have the same resolution
        if len(set(zip(widths, heights))) > 1:
            for cap in video_caps:
                cap.release()
            raise ValueError("All videos must have the same resolution (width and height).")

        # Use first video's properties for output
        out_fps = fps_list[0]
        out_width, out_height = widths[0], heights[0]
        expected_duration = clip_duration * num_videos

        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))
        if not out.isOpened():
            for cap in video_caps:
                cap.release()
            raise ValueError("Failed to initialize output video writer.")

        # Process each video clip
        for i, cap in enumerate(video_caps):
            start_time = i * clip_duration
            start_frame = int(start_time * fps_list[i])
            end_frame = int((start_time + clip_duration) * fps_list[i])

            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read and write frames
            for _ in range(start_frame, min(end_frame, frame_counts[i])):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()

        out.release()

        # Verify final video duration
        final_cap = cv2.VideoCapture(output_path)
        if not final_cap.isOpened():
            raise ValueError("Failed to open output video for verification.")
        final_frame_count = int(final_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        final_fps = final_cap.get(cv2.CAP_PROP_FPS)
        final_duration = final_frame_count / final_fps if final_fps > 0 else 0
        final_cap.release()

        if abs(final_duration - expected_duration) > 0.5:
            raise ValueError(f"Final video duration ({final_duration}s) does not match expected ({expected_duration}s).")
        
    def post(self, request):
        try:
            video_uuid = request.session.get('video_uuid')
            if not video_uuid:
                return JsonResponse({"error": "No video UUID found in session"}, status=400)

            # Use MEDIA_ROOT for file system paths
            video_paths = [
                os.path.join(settings.MEDIA_ROOT, f"output/posture_video_{video_uuid}.mp4"),
                os.path.join(settings.MEDIA_ROOT, f"output/eye_analysis_video_{video_uuid}.mp4"),
            ]

            # Verify input files exist
            for video_path in video_paths:
                if not os.path.exists(video_path):
                    return JsonResponse({"error": f"Input video not found: {os.path.basename(video_path)}"}, status=404)
                
            # Create output directory if it doesn't exist
            output_dir = os.path.join(settings.MEDIA_ROOT, "merged")
            os.makedirs(output_dir, exist_ok=True)

            original_video = os.path.join(settings.MEDIA_ROOT, f"videos/video_{video_uuid}.mp4")

            temp_video = os.path.join(settings.MEDIA_ROOT, "merged/merged_output_temp.mp4")
            encoded_video = os.path.join(settings.MEDIA_ROOT, "merged/encoded_merged_output.mp4")
            output_with_audio = os.path.join(settings.MEDIA_ROOT, "merged/final_output_with_audio.mp4")

            # Perform video processing
           # Perform video processing
            try:
                print("Merging videos...")
                self.merge_videos(video_paths, temp_video)

                print("Encoding merged video...")
                encoded_video_path = self.encode_video(temp_video, encoded_video)

                print("Adding audio...")
                final_video_path = self.add_audio(encoded_video_path, original_video, output_with_audio)
                print(f"Final video with audio saved as: {final_video_path}")

            except Exception as e:
                print(f"Video processing failed: {e}")
                return JsonResponse({"error": f"Video processing failed: {str(e)}"}, status=500)


            # Return URL (not file system path) for the client
            final_video_url = f"{settings.MEDIA_URL}merged/final_output_with_audio.mp4"

            return JsonResponse({
                'video_path': final_video_url,
            })

        except Exception as e:
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)

class Home(View):
    def get(self, request):
        try:
            
            video_count = VideoCount.objects.get_or_create(id=1)[0].count

            # Clear only audio file if it exists
            if 'extracted_audio_path' in request.session:
                try:
                    # audio_path = request.session['extracted_audio_path']
                    # full_path = os.path.join(settings.MEDIA_ROOT, audio_path)
                    # if os.path.exists(full_path):
                    #     os.remove(full_path)
                    #     #print(f"Removed audio file: {full_path}")
                    del request.session['extracted_audio_path']
                except Exception as e:
                    print(f"Error removing audio file: {str(e)}")
            return render(request, "index.html", {'video_count': video_count})
        except Exception as e:
            #print(f"Error getting video count: {e}")
            return render(request, "index.html", {'video_count': 0})
# -------------------- video uploading class --------------------
class UploadVideo(View):
    def post(self, request):
        try:
            # Validate request
            if not request.FILES:
                return JsonResponse({"error": "No file uploaded"}, status=400)
            
            video_file = request.FILES.get('video-file')
            if not video_file:
                return JsonResponse({"error": "No video file uploaded."}, status=400)
            
            # Update view counter
            video_counter, created = VideoCount.objects.get_or_create(id=1)
            VideoCount.objects.filter(id=1).update(count=F('count') + 1)

            video_uuid = str(uuid.uuid4())
            request.session['video_uuid'] = video_uuid
            
            # Clear previous files if they exist
            for key in ['uploaded_video_path', 'extracted_audio_path']:
                if key in request.session:
                    try:
                        old_path = request.session[key]
                        full_path = os.path.join(settings.MEDIA_ROOT, old_path)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            #print(f"Removed old file: {full_path}")
                    except Exception as e:
                        print(f"Error removing old file: {str(e)}")
                    del request.session[key]

            # Save new video
            video_path = save_uploaded_video(request,video_file)
            if not video_path:
                return JsonResponse({"error": "Failed to save video"}, status=500)
            
            # Store video path in session
            request.session["uploaded_video_path"] = video_path

            # Extract audio using standalone function
            audio_path = extract_and_store_audio(video_path, request)
            if not audio_path:
                return JsonResponse({"error": "Failed to extract audio"}, status=500)

            return JsonResponse({
                "success": True, 
                "message": "Video uploaded and audio extracted successfully",
                "file_path": video_path
            })
            
        except Exception as e:
            #print(f"Error uploading video: {str(e)}")
            return JsonResponse({"error": str(e)}, status=500)