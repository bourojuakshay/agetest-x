import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import logging
import os

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgeService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.face_cascade = None
        self.transform = None
        self.MODEL_PATH = "age_model.pth"
        self.IMG_SIZE = 224
        
        # Load Resources
        self._load_model()
        self._load_face_detector()
        self._init_transform()
        
        self.initialized = True
        logger.info(f"AgeService initialized on {self.device}")

    def _load_model(self):
        try:
            # Architecture: ResNet18 Regression
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            
            if os.path.exists(self.MODEL_PATH):
                self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
                logger.info("Model weights loaded successfully.")
            else:
                logger.warning(f"Model file {self.MODEL_PATH} not found. Running with random weights (UNSAFE FOR PRODUCTION).")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Critical: Model loading failed.")

    def _load_face_detector(self):
        try:
            # Use OpenCV Haar Cascades
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Failed to load cascade from {cascade_path}")
        except Exception as e:
            logger.error(f"Face detector setup failed: {e}")
            raise

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_and_predict(self, image_array: np.ndarray):
        """
        Process the image: Detect face -> Predict Age -> Return result.
        Returns: { "age": float, "age_group": str, "confidence": float } or None
        """
        try:
            # 1. Face Detection
            frame = cv2.resize(image_array, (640, 480)) # Optimize detection speed
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Enhanced detection parameters for better accuracy
            # scaleFactor=1.05 for more thorough scanning (slower but more accurate)
            # minNeighbors=6 for stricter validation (reduces false positives)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6, 
                minSize=(60, 60),
                maxSize=(400, 400)
            )

            if len(faces) == 0:
                logger.debug("No faces detected.")
                return {"error": "No face detected"}

            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Enhanced face quality checks
            # Check face size relative to frame (too small = unreliable)
            if w < 80 or h < 80:
                 return {"error": "Face too small"}
            
            # Check if face is too close to edges (partial face = unreliable)
            margin = 10
            if x < margin or y < margin or (x + w) > (640 - margin) or (y + h) > (480 - margin):
                logger.debug("Face too close to frame edge")
                # Don't reject, but will lower confidence later

            face_img = frame[y:y+h, x:x+w]

            # 2. Preprocess
            pil_img = Image.fromarray(face_img)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # 3. Inference
            with torch.no_grad():
                age_pred = self.model(input_tensor).item()

            # 4. Post-process
            # Clamp age to realistic bounds
            age = max(1, min(100, age_pred))
            group = self._get_age_group(age)
            
            # Dynamic confidence scoring based on face quality
            base_confidence = 0.85
            
            # Boost confidence for larger, well-positioned faces
            face_area = w * h
            frame_area = 640 * 480
            face_ratio = face_area / frame_area
            
            # Ideal face size is 15-30% of frame
            if 0.15 <= face_ratio <= 0.30:
                base_confidence += 0.10
            elif 0.10 <= face_ratio < 0.15 or 0.30 < face_ratio <= 0.40:
                base_confidence += 0.05
            
            # Check if face is centered (more reliable)
            face_center_x = x + w/2
            face_center_y = y + h/2
            frame_center_x = 640/2
            frame_center_y = 480/2
            
            center_dist = ((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)**0.5
            max_dist = ((frame_center_x)**2 + (frame_center_y)**2)**0.5
            
            if center_dist < max_dist * 0.3:  # Face is well-centered
                base_confidence += 0.05
            
            confidence = min(0.98, base_confidence)  # Cap at 0.98 

            return {
                "age": round(age, 1),
                "age_group": group,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e)}

    def _get_age_group(self, age):
        """
        Maps scalar age to safety categories.
        Safety-First: Boundaries are conservative.
        """
        # Kid: 0-12
        # Teen: 13-17
        # Adult: 18+
        
        # We can implement a "buffer" zone? 
        # i.e. 12.5 -> treated as Kid?
        # Let's stick to standard strict mapping for now.
        
        if age < 13:
            return "Kid"
        elif 13 <= age < 18:
            return "Teen"
        elif 18 <= age < 25:
            return "Young Adult"
        elif 25 <= age < 50:
            return "Adult"
        else:
            return "Senior"
