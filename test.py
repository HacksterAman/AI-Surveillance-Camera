import cv2
import glob
import os
import os.path as osp
import onnxruntime
import numpy as np
import time
from collections import defaultdict, deque
from numpy.linalg import norm as l2norm

# Hardcoded Constants
MODEL_NAME = 'buffalo_m'  # Recommended starting model - balanced speed/accuracy
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# TensorRT Provider Configuration
PROVIDERS = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': 'trt_cache'
    })
]

ALLOWED_MODULES = ['detection', 'genderage', 'recognition']
DETECTION_SIZE = (640, 640)


class Face(dict):
    """Face detection result with embedding support"""
    
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
        
        # Reset cached properties when embedding changes
        if name == 'embedding':
            self._reset_cached_properties()
        
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
            self._embedding_norm = l2norm(self.embedding)
        return self._embedding_norm

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        if not hasattr(self, '_normed_embedding') or self._normed_embedding is None:
            norm = self.embedding_norm
            if norm > 0:
                self._normed_embedding = self.embedding / norm
            else:
                self._normed_embedding = self.embedding
        return self._normed_embedding
    
    def _reset_cached_properties(self):
        """Reset cached embedding properties when embedding changes"""
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
    """Face embedding averaging system for 2-second windows"""
    
    def __init__(self, avg_window_seconds=2.0, similarity_threshold=0.6, max_missing_frames=30):
        self.avg_window_seconds = avg_window_seconds
        self.similarity_threshold = similarity_threshold  # Kept for compatibility but not used
        self.max_missing_frames = max_missing_frames  # Kept for compatibility but not used
        
        # Face embedding data
        self.face_data = {}  # face_key -> face tracking info
        self.frame_count = 0
        
        # Timing
        self.last_cleanup = time.time()
        self.cleanup_interval = 5.0  # seconds
    
    
    def update_face_data(self, face_key, face, current_time):
        """Update face data with new detection for embedding averaging"""
        if face_key not in self.face_data:
            self.face_data[face_key] = {
                'embeddings': deque(),
                'timestamps': deque(),
                'avg_embedding': None,
                'avg_norm': None,
                'last_seen': current_time
            }
        
        face_info = self.face_data[face_key]
        
        # Add new embedding if available
        if hasattr(face, 'embedding') and face.embedding is not None:
            face_info['embeddings'].append(face.embedding.copy())
            face_info['timestamps'].append(current_time)
            
            # Remove old embeddings outside the window
            cutoff_time = current_time - self.avg_window_seconds
            while (face_info['timestamps'] and 
                   face_info['timestamps'][0] < cutoff_time):
                face_info['embeddings'].popleft()
                face_info['timestamps'].popleft()
            
            # Calculate average embedding
            if face_info['embeddings']:
                embeddings_array = np.array(list(face_info['embeddings']))
                face_info['avg_embedding'] = np.mean(embeddings_array, axis=0)
                face_info['avg_norm'] = l2norm(face_info['avg_embedding'])
        
        # Update timestamp
        face_info['last_seen'] = current_time
    
    def track_faces(self, faces):
        """Apply 2-second embedding averaging to detected faces"""
        current_time = time.time()
        self.frame_count += 1
        
        # For each detected face, simply apply 2-second averaging
        tracked_faces = []
        
        for i, face in enumerate(faces):
            # Use face index as simple identifier for this frame
            face_key = f"face_{i}"
            
            # Update face data (without persistent ID tracking)
            self.update_face_data(face_key, face, current_time)
            
            # Create tracked face object
            tracked_face = Face()
            for key, value in face.__dict__.items():
                setattr(tracked_face, key, value)
            
            # Add averaged embedding data if available
            face_info = self.face_data.get(face_key)
            if face_info and face_info['avg_embedding'] is not None:
                tracked_face.avg_embedding = face_info['avg_embedding']
                tracked_face.avg_embedding_norm = face_info['avg_norm']
            
            tracked_faces.append(tracked_face)
        
        # Periodic cleanup of old face data
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_faces(current_time)
            self.last_cleanup = current_time
        
        return tracked_faces
    
    def cleanup_old_faces(self, current_time):
        """Remove old face data to prevent memory leaks"""
        cutoff_time = current_time - (self.avg_window_seconds * 3)  # Keep 3x window for safety
        
        faces_to_remove = []
        for face_key, face_info in self.face_data.items():
            if face_info['last_seen'] < cutoff_time:
                faces_to_remove.append(face_key)
        
        for face_key in faces_to_remove:
            del self.face_data[face_key]
        
        if faces_to_remove:
            print(f"Cleaned up {len(faces_to_remove)} old face records")


def get_model_path():
    """Get model directory path with error handling"""
    try:
        if not osp.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        print(f"ERROR: Failed to locate models - {e}")
        raise


def load_model(onnx_file, **kwargs):
    """Load ONNX model with error handling"""
    try:
        from insightface.model_zoo import get_model
        return get_model(onnx_file, **kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load model {onnx_file} - {e}")
        return None


class FaceAnalysis:
    """Streamlined face analysis with embedded models"""
    
    def __init__(self, allowed_modules=None, **kwargs):
        self.models = {}
        self.det_model = None
        
        try:
            # Suppress ONNX runtime warnings
            onnxruntime.set_default_logger_severity(3)
            
            # Get model directory
            self.model_dir = get_model_path()
            
            # Set allowed modules
            if allowed_modules is None:
                allowed_modules = ALLOWED_MODULES
            
            # Load models
            self._load_models(allowed_modules, **kwargs)
            
            # Ensure detection model exists
            if 'detection' not in self.models:
                raise RuntimeError("Detection model is required but not found")
            
            self.det_model = self.models['detection']
            
        except Exception as e:
            print(f"ERROR: FaceAnalysis initialization failed - {e}")
            raise

    def _load_models(self, allowed_modules, **kwargs):
        """Load and validate ONNX models"""
        try:
            onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
            if not onnx_files:
                raise FileNotFoundError(f"No ONNX models found in {self.model_dir}")
            
            onnx_files = sorted(onnx_files)
            loaded_count = 0
            
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
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"WARNING: Failed to process {onnx_file} - {e}")
                    continue
            
            if loaded_count == 0:
                raise RuntimeError("No models were successfully loaded")
                
        except Exception as e:
            print(f"ERROR: Model loading failed - {e}")
            raise

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=None):
        """Prepare models for inference"""
        try:
            if det_size is None:
                det_size = DETECTION_SIZE
            
            self.det_thresh = det_thresh
            self.det_size = det_size
            
            for taskname, model in self.models.items():
                try:
                    if taskname == 'detection':
                        model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
                    else:
                        model.prepare(ctx_id)
                except Exception as e:
                    print(f"ERROR: Failed to prepare {taskname} model - {e}")
                    raise
                    
        except Exception as e:
            print(f"ERROR: Model preparation failed - {e}")
            raise

    def get(self, img, max_num=0):
        """Extract faces with embeddings and attributes"""
        try:
            if self.det_model is None:
                raise RuntimeError("Detection model not available")
            
            # Detect faces
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
                    
                    # Extract features from other models
                    for taskname, model in self.models.items():
                        if taskname == 'detection':
                            continue
                        try:
                            model.get(img, face)
                        except Exception as e:
                            print(f"WARNING: {taskname} extraction failed for face {i} - {e}")
                            continue
                    
                    faces.append(face)
                    
                except Exception as e:
                    print(f"WARNING: Face {i} processing failed - {e}")
                    continue
            
            return faces
            
        except Exception as e:
            print(f"ERROR: Face detection failed - {e}")
            return []

    def draw_on(self, img, faces):
        """Draw detection results on image with averaged embeddings"""
        try:
            if img is None:
                raise ValueError("Input image is None")
            
            dimg = img.copy()
            
            for i, face in enumerate(faces):
                try:
                    box = face.bbox.astype(int)
                    
                    # Use default red color for bounding box
                    color = (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    # Draw keypoints
                    if face.kps is not None:
                        kps = face.kps.astype(int)
                        for l in range(kps.shape[0]):
                            kp_color = (0, 255, 0) if l in [0, 3] else (0, 0, 255)
                            cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, kp_color, 2)
                    
                    # Draw age and gender
                    if face.gender is not None and face.age is not None:
                        gender = "Male" if face.gender == 1 else "Female"
                        text = f"{gender}, {int(face.age)}"
                        cv2.putText(dimg, text, (box[0], box[1] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Draw embedding info (averaged over 2 seconds)
                    y_offset = box[3] + 20
                    
                    # Show averaged embedding norm if available
                    if hasattr(face, 'avg_embedding_norm') and face.avg_embedding_norm is not None:
                        avg_norm_text = f"Avg Emb: {face.avg_embedding_norm:.3f}"
                        cv2.putText(dimg, avg_norm_text, (box[0], y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 15
                    
                    # Show instant embedding norm for comparison (in smaller text)
                    if hasattr(face, 'embedding') and face.embedding is not None:
                        inst_norm = f"Inst: {face.embedding_norm:.3f}" if face.embedding_norm else "Inst: N/A"
                        cv2.putText(dimg, inst_norm, (box[0], y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                        
                except Exception as e:
                    print(f"WARNING: Failed to draw face {i} - {e}")
                    continue
            
            return dimg
            
        except Exception as e:
            print(f"ERROR: Drawing failed - {e}")
            return img


# Initialize face analysis system
try:
    app = FaceAnalysis(allowed_modules=ALLOWED_MODULES, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    
    # Initialize face tracker for 2-second embedding averaging
    face_tracker = FaceTracker(avg_window_seconds=2.0, similarity_threshold=0.6, max_missing_frames=30)
    print("Embedding averaging system initialized with 2-second window")
    
except Exception as e:
    print(f"CRITICAL ERROR: System initialization failed - {e}")
    exit(1)


if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("AI SURVEILLANCE CAMERA - EMBEDDING AVERAGING SYSTEM")
        print("="*60)
        print(f"✓ Models loaded: {list(app.models.keys())}")
        print(f"✓ 2-second embedding averaging for stable measurements")
        print(f"✓ Real-time FPS monitoring")
        print(f"✓ Enhanced face detection with attributes")
        print("="*60)
        print("Controls: Press 'q' to quit")
        print("="*60)
        
        # Open webcam
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            print("Trying webcam index 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("ERROR: Could not open any webcam")
                exit(1)

        # FPS calculation variables
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                break

            try:
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                elapsed_time = current_time - fps_start_time
                
                if elapsed_time >= 1.0:  # Update FPS every second
                    current_fps = fps_counter / elapsed_time
                    fps_counter = 0
                    fps_start_time = current_time
                
                # Detect faces
                detected_faces = app.get(frame)
                
                # Track faces with 2-second averaged embeddings
                tracked_faces = face_tracker.track_faces(detected_faces)
                
                # Draw results with averaged embeddings
                frame_out = app.draw_on(frame, tracked_faces)
                
                # Add FPS overlay
                fps_text = f"FPS: {current_fps:.1f} | Faces: {len(tracked_faces)}"
                cv2.putText(frame_out, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame with updated title
                cv2.imshow("AI Surveillance - 2s Averaged Embeddings", frame_out)
                
            except Exception as e:
                print(f"ERROR: Frame processing failed - {e}")
                frame_out = frame
                cv2.imshow("AI Surveillance - 2s Averaged Embeddings", frame_out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"CRITICAL ERROR: Application failed - {e}")
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass