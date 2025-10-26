import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from numpy.linalg import norm as l2norm

import os
import os.path as osp
import glob
import onnxruntime
MODEL_DIR = 'models'
ALLOWED_MODULES = ['detection', 'genderage', 'recognition']
DETECTION_SIZE = (640, 640)
EPSILON = 1e-12

MODEL_OPTIONS = {
    0: {'name': 'buffalo_s', 'label': 'Speed', 'precision': 'float32'},
    1: {'name': 'buffalo_l', 'label': 'Accuracy', 'precision': 'float64'}
}

RESOLUTION_OPTIONS = {
    0: {'size': (320, 320), 'label': '320p'},
    1: {'size': (640, 640), 'label': '640p'},
    2: {'size': (1024, 1024), 'label': '1024p'}
}

def get_providers(use_fp16=False):
    return [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': use_fp16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'trt_cache',
            'trt_int8_enable': False,
        }),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]


class Face(dict):
    high_precision_mode = True
    
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        
        if name == 'embedding' and value is not None:
            self._reset_cached_properties()
            if hasattr(value, 'dtype') and hasattr(value, 'astype'):
                dtype = np.float64 if Face.high_precision_mode else np.float32
                value = value.astype(dtype)
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    value = None
        
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        if not hasattr(self, '_embedding_norm') or self._embedding_norm is None:
            dtype = np.float64 if Face.high_precision_mode else np.float32
            self._embedding_norm = l2norm(self.embedding.astype(dtype))
        return self._embedding_norm

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        if not hasattr(self, '_normed_embedding') or self._normed_embedding is None:
            norm = self.embedding_norm
            dtype = np.float64 if Face.high_precision_mode else np.float32
            if norm > EPSILON:
                self._normed_embedding = (self.embedding / (norm + EPSILON)).astype(dtype)
            else:
                self._normed_embedding = np.zeros_like(self.embedding, dtype=dtype)
        return self._normed_embedding
    
    def compute_similarity(self, other_face):
        if self.embedding is None or other_face.embedding is None:
            return None
        norm1 = self.normed_embedding
        norm2 = other_face.normed_embedding
        if norm1 is None or norm2 is None:
            return None
        similarity = np.dot(norm1, norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def _reset_cached_properties(self):
        if hasattr(self, '_embedding_norm'):
            self._embedding_norm = None
        if hasattr(self, '_normed_embedding'):
            self._normed_embedding = None

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


class FaceTracker:
    def __init__(self, avg_window_seconds=1.0):
        self.avg_window_seconds = avg_window_seconds
        self.face_data = {}
        self.frame_count = 0
        self.current_period_start = time.time()
        self.current_period_embeddings = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 5.0
        self.high_precision_mode = Face.high_precision_mode
    
    def update_face_data(self, face_key, face, current_time):
        if face_key not in self.face_data:
            self.face_data[face_key] = {
                'avg_embedding': None,
                'avg_norm': None,
                'last_seen': current_time,
                'welford_count': 0,
                'welford_mean': None,
                'welford_m2': None
            }
        
        face_info = self.face_data[face_key]
        
        if hasattr(face, 'embedding') and face.embedding is not None:
            if np.any(np.isnan(face.embedding)) or np.any(np.isinf(face.embedding)):
                return
            
            embedding = face.embedding.copy()
            if hasattr(embedding, 'dtype') and hasattr(embedding, 'astype'):
                dtype = np.float64 if Face.high_precision_mode else np.float32
                embedding = embedding.astype(dtype)
            
            if face_key not in self.current_period_embeddings:
                self.current_period_embeddings[face_key] = []
            
            self.current_period_embeddings[face_key].append(embedding)
        
        face_info['last_seen'] = current_time
    
    def check_and_update_averages(self, current_time):
        if current_time - self.current_period_start >= self.avg_window_seconds:
            for face_key, embeddings in self.current_period_embeddings.items():
                if embeddings and face_key in self.face_data:
                    dtype = np.float64 if Face.high_precision_mode else np.float32
                    count = len(embeddings)
                    mean = np.zeros_like(embeddings[0], dtype=dtype)
                    m2 = np.zeros_like(embeddings[0], dtype=dtype)
                    
                    for i, embedding in enumerate(embeddings, start=1):
                        delta = embedding - mean
                        mean += delta / i
                        delta2 = embedding - mean
                        m2 += delta * delta2
                    
                    avg_embedding = mean.astype(dtype)
                    
                    if np.any(np.isnan(avg_embedding)) or np.any(np.isinf(avg_embedding)):
                        continue
                    
                    avg_norm = l2norm(avg_embedding)
                    
                    self.face_data[face_key]['avg_embedding'] = avg_embedding
                    self.face_data[face_key]['avg_norm'] = avg_norm
                    self.face_data[face_key]['welford_count'] = count
                    self.face_data[face_key]['welford_mean'] = mean
                    self.face_data[face_key]['welford_m2'] = m2
            
            self.current_period_start = current_time
            self.current_period_embeddings = {}
    
    def track_faces(self, faces):
        current_time = time.time()
        self.frame_count += 1
        
        self.check_and_update_averages(current_time)
        
        tracked_faces = []
        
        for i, face in enumerate(faces):
            face_key = f"face_{i}"
            self.update_face_data(face_key, face, current_time)
            
            tracked_face = Face()
            for key, value in face.__dict__.items():
                setattr(tracked_face, key, value)
            
            face_info = self.face_data.get(face_key)
            if face_info and face_info['avg_embedding'] is not None:
                dtype = np.float64 if Face.high_precision_mode else np.float32
                tracked_face.avg_embedding = face_info['avg_embedding'].astype(dtype)
                tracked_face.avg_embedding_norm = face_info['avg_norm']
                tracked_face.avg_embedding_count = face_info.get('welford_count', 0)
            
            tracked_faces.append(tracked_face)
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_faces(current_time)
            self.last_cleanup = current_time
        
        return tracked_faces
    
    def cleanup_old_faces(self, current_time):
        cutoff_time = current_time - (self.avg_window_seconds * 5)
        faces_to_remove = [k for k, v in self.face_data.items() if v['last_seen'] < cutoff_time]
        for face_key in faces_to_remove:
            del self.face_data[face_key]


def load_model(onnx_file, **kwargs):
    try:
        from insightface.model_zoo import get_model
        return get_model(onnx_file, **kwargs)
    except Exception:
        return None


class FaceAnalysis:
    def __init__(self, model_name='buffalo_l', allowed_modules=None, **kwargs):
        self.models = {}
        self.det_model = None
        self.current_model = model_name
        
        onnxruntime.set_default_logger_severity(3)
        
        model_path = os.path.join(MODEL_DIR, model_name)
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        self.model_dir = model_path
        
        if allowed_modules is None:
            allowed_modules = ALLOWED_MODULES
        
        self._load_models(allowed_modules, **kwargs)
        
        if 'detection' not in self.models:
            raise RuntimeError("Detection model is required")
        
        self.det_model = self.models['detection']

    def _load_models(self, allowed_modules, **kwargs):
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX models found in {self.model_dir}")
        
        onnx_files = sorted(onnx_files)
        
        for onnx_file in onnx_files:
            try:
                model = load_model(onnx_file, **kwargs)
                if model is None:
                    continue
                if model.taskname not in allowed_modules:
                    del model
                    continue
                if model.taskname in self.models:
                    del model
                    continue
                self.models[model.taskname] = model
            except Exception:
                continue

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=None):
        if det_size is None:
            det_size = DETECTION_SIZE
        
        self.det_thresh = det_thresh
        self.det_size = det_size
        
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        if self.det_model is None:
            return []
        
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        
        if bboxes.shape[0] == 0:
            return []
        
        faces = []
        for i in range(bboxes.shape[0]):
            try:
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = kpss[i] if kpss is not None else None
                
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                
                for taskname, model in self.models.items():
                    if taskname == 'detection':
                        continue
                    try:
                        model.get(img, face)
                    except Exception:
                        continue
                
                faces.append(face)
            except Exception:
                continue
        
        return faces


class SurveillanceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Surveillance Camera - High Precision Mode")
        self.setGeometry(100, 100, 1600, 900)
        
        # State variables
        self.camera_index = 0
        self.cap = None
        self.running = False
        self.faces = []
        self.fps = 0
        self.current_model_index = 1  # Start with Accuracy (buffalo_l)
        self.current_resolution_index = 1  # Start with 640p
        self.use_high_precision = True
        
        # Initialize models
        try:
            self.load_model(self.current_model_index)
        except Exception as e:
            print(f"Failed to initialize: {e}")
            sys.exit(1)
        
        self.setup_ui()
        self.apply_stylesheet()
        self.start_camera()
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
    
    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        
        # Top control bar
        control_bar = QWidget()
        control_bar.setObjectName("controlBar")
        control_bar.setFixedHeight(70)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(15, 10, 15, 10)
        control_bar.setLayout(control_layout)
        
        self.camera_btn = QPushButton(f"📹 Camera {self.camera_index}")
        self.camera_btn.setObjectName("cameraButton")
        self.camera_btn.setFixedSize(150, 50)
        self.camera_btn.clicked.connect(self.switch_camera)
        control_layout.addWidget(self.camera_btn)
        
        control_layout.addSpacing(20)
        
        self.model_btn = QPushButton("🎯 Accuracy")
        self.model_btn.setObjectName("modelButton")
        self.model_btn.setFixedSize(150, 50)
        self.model_btn.clicked.connect(self.toggle_model)
        control_layout.addWidget(self.model_btn)
        
        control_layout.addSpacing(20)
        
        self.resolution_btn = QPushButton("📐 640p")
        self.resolution_btn.setObjectName("resolutionButton")
        self.resolution_btn.setFixedSize(150, 50)
        self.resolution_btn.clicked.connect(self.toggle_resolution)
        control_layout.addWidget(self.resolution_btn)
        
        control_layout.addStretch()
        
        # Stats label
        self.stats_label = QLabel("FPS: 0 | Faces: 0 | Precision: float64")
        self.stats_label.setObjectName("statsLabel")
        control_layout.addWidget(self.stats_label)
        
        main_layout.addWidget(control_bar)
        
        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        
        # Left: Video feed
        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(5, 5, 5, 5)
        video_frame.setLayout(video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: #000000;")
        video_layout.addWidget(self.video_label)
        
        content_layout.addWidget(video_frame, stretch=7)
        
        # Right: Embeddings panel
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_frame.setFixedWidth(480)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_frame.setLayout(info_layout)
        
        # Header
        header_label = QLabel("FACE EMBEDDINGS")
        header_label.setObjectName("headerLabel")
        header_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(header_label)
        
        # Text area for embeddings
        self.info_text = QTextEdit()
        self.info_text.setObjectName("infoText")
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        content_layout.addWidget(info_frame, stretch=3)
        
        main_layout.addLayout(content_layout)
    
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            
            #controlBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #1f1f1f);
                border-radius: 10px;
                border: 1px solid #3a3a3a;
            }
            
            #cameraButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
            
            #cameraButton:hover {
                background-color: #4a4a4a;
                border: 2px solid #5a5a5a;
            }
            
            #cameraButton:pressed {
                background-color: #2a2a2a;
            }
            
            QPushButton:pressed {
                padding-top: 7px;
                padding-left: 7px;
            }
            
            #statsLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
            
            #videoFrame {
                background-color: #0a0a0a;
                border: 2px solid #3a3a3a;
                border-radius: 10px;
            }
            
            #infoFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #252525, stop:1 #1a1a1a);
                border: 2px solid #3a3a3a;
                border-radius: 10px;
            }
            
            #headerLabel {
                color: #00ccff;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            
            #infoText {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                selection-background-color: #0078d7;
            }
            
            #modelButton {
                background: #00dd00;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            
            #modelButton:hover {
                background: #00ff00;
            }
            
            #modelButton:pressed {
                background: #009900;
            }
            
            #resolutionButton {
                background: #0088dd;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            
            #resolutionButton:hover {
                background: #00aaff;
            }
            
            #resolutionButton:pressed {
                background: #0066aa;
            }
        """)
    
    def load_model(self, model_index, resolution_index=None):
        if resolution_index is None:
            resolution_index = self.current_resolution_index
        
        model_info = MODEL_OPTIONS[model_index]
        resolution_info = RESOLUTION_OPTIONS[resolution_index]
        model_name = model_info['name']
        det_size = resolution_info['size']
        
        print(f"Loading {model_name} model with {resolution_info['label']} resolution...")
        
        try:
            providers = get_providers(use_fp16=False)
            self.app = FaceAnalysis(model_name=model_name, allowed_modules=ALLOWED_MODULES, providers=providers)
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.face_tracker = FaceTracker(avg_window_seconds=1.0)
            
            # Set precision based on model
            if model_info['precision'] == 'float32':
                Face.high_precision_mode = False
                self.use_high_precision = False
            else:
                Face.high_precision_mode = True
                self.use_high_precision = True
            
            print(f"✓ {model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def toggle_model(self):
        new_index = 1 - self.current_model_index
        self.running = False
        if self.cap:
            self.cap.release()
        
        try:
            self.load_model(new_index)
            self.current_model_index = new_index
            
            # Speed model uses float32, Accuracy uses float64
            if new_index == 0:  # Speed
                self.model_btn.setText("⚡ Speed")
                Face.high_precision_mode = False
                self.use_high_precision = False
            else:  # Accuracy
                self.model_btn.setText("🎯 Accuracy")
                Face.high_precision_mode = True
                self.use_high_precision = True
            
            # Update stats display
            precision = 'float32' if new_index == 0 else 'float64'
            model_name = MODEL_OPTIONS[new_index]['label']
            
            self.start_camera()
            print(f"Switched to {model_name} ({precision})")
        except Exception as e:
            print(f"Failed: {e}")
    
    def toggle_resolution(self):
        self.running = False
        if self.cap:
            self.cap.release()
        
        # Cycle through resolution options
        self.current_resolution_index = (self.current_resolution_index + 1) % len(RESOLUTION_OPTIONS)
        resolution_info = RESOLUTION_OPTIONS[self.current_resolution_index]
        
        try:
            self.load_model(self.current_model_index, self.current_resolution_index)
            self.resolution_btn.setText(f"📐 {resolution_info['label']}")
            
            self.start_camera()
            print(f"Switched to {resolution_info['label']} resolution")
        except Exception as e:
            print(f"Failed to switch resolution: {e}")
    
    def switch_camera(self):
        if self.cap:
            self.cap.release()
        
        self.camera_index = (self.camera_index + 1) % 5
        self.start_camera()
        self.camera_btn.setText(f"📹 Camera {self.camera_index}")
    
    def start_camera(self):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            if self.camera_index != 0:
                self.camera_index = 0
                self.cap = cv2.VideoCapture(0)
                self.camera_btn.setText(f"📹 Camera {self.camera_index}")
        
        self.running = self.cap.isOpened()
    
    def update_frame(self):
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        start_time = time.time()
        
        # Detect faces
        detected_faces = self.app.get(frame)
        tracked_faces = self.face_tracker.track_faces(detected_faces)
        self.faces = tracked_faces
        
        # Draw faces on frame
        frame = self.draw_faces(frame, tracked_faces)
        
        # Calculate FPS
        self.fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        # Update stats
        dtype = "float64" if Face.high_precision_mode else "float32"
        model_info = MODEL_OPTIONS[self.current_model_index]
        self.stats_label.setText(f"FPS: {self.fps:.1f} | Faces: {len(tracked_faces)} | Model: {model_info['label']} | {dtype}")
        
        # Convert frame for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update info panel
        self.update_info_panel(tracked_faces)
    
    def draw_faces(self, img, faces):
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw keypoints
            if face.kps is not None:
                kps = face.kps.astype(int)
                for l in range(kps.shape[0]):
                    kp_color = (0, 255, 0) if l in [0, 3] else (0, 0, 255)
                    cv2.circle(img, (kps[l][0], kps[l][1]), 2, kp_color, -1)
            
            # Draw age and gender
            if face.gender is not None and face.age is not None:
                gender = "Male" if face.gender == 1 else "Female"
                text = f"{gender}, {int(face.age)}"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (box[0], box[1] - text_height - 10), 
                             (box[0] + text_width + 10, box[1]), (0, 0, 0), -1)
                
                cv2.putText(img, text, (box[0] + 5, box[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return img
    
    def update_info_panel(self, faces):
        # Save current scroll position
        scrollbar = self.info_text.verticalScrollBar()
        scroll_position = scrollbar.value()
        
        html = """
        <style>
            body { background-color: #1a1a1a; color: #e0e0e0; font-family: Consolas, monospace; }
            .face-header { color: #00ff00; font-weight: bold; font-size: 12px; margin-top: 10px; }
            .emb-header { color: #ffaa00; font-weight: bold; margin-top: 8px; }
            .emb-data { color: #cccccc; margin-left: 10px; }
            .info { color: #999999; margin-left: 10px; }
            .separator { color: #3a3a3a; }
        </style>
        """
        
        if not faces:
            html += "<p style='color: #666666; text-align: center; margin-top: 50px;'>No faces detected</p>"
        else:
            for i, face in enumerate(faces, 1):
                html += f"<div class='face-header'>═══ FACE #{i} ═══</div>"
                
                # Raw Embedding
                if face.embedding is not None:
                    html += "<div class='emb-header'>● Raw Embedding (512D):</div>"
                    emb_str = ", ".join([f"{v:.6f}" for v in face.embedding[:10]])
                    html += f"<div class='emb-data'>[{emb_str}, ...]</div>"
                    html += f"<div class='info'>Norm: {face.embedding_norm:.6f}</div>"
                    html += f"<div class='info'>Type: {face.embedding.dtype}</div>"
                
                # Normalized embedding
                if face.normed_embedding is not None:
                    html += "<div class='emb-header'>● Normalized Embedding:</div>"
                    norm_str = ", ".join([f"{v:.6f}" for v in face.normed_embedding[:10]])
                    html += f"<div class='emb-data'>[{norm_str}, ...]</div>"
                    html += f"<div class='info'>Norm: {l2norm(face.normed_embedding):.6f}</div>"
                
                # Averaged embedding
                if hasattr(face, 'avg_embedding') and face.avg_embedding is not None:
                    html += "<div class='emb-header'>● 1s Averaged Embedding:</div>"
                    avg_str = ", ".join([f"{v:.6f}" for v in face.avg_embedding[:10]])
                    html += f"<div class='emb-data'>[{avg_str}, ...]</div>"
                    html += f"<div class='info'>Avg Norm: {face.avg_embedding_norm:.6f}</div>"
                    if hasattr(face, 'avg_embedding_count'):
                        html += f"<div class='info'>Samples: {face.avg_embedding_count}</div>"
                
                html += "<div class='separator'>═════════════════════════════════════════════════</div>"
        
        self.info_text.setHtml(html)
        
        # Restore scroll position
        scrollbar.setValue(scroll_position)
    
    def closeEvent(self, event):
        self.running = False
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = SurveillanceApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

