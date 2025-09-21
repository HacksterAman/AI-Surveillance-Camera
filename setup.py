#!/usr/bin/env python3
"""
AI Surveillance Camera v1.0 Setup Script
This script helps users download and set up InsightFace models automatically.
"""

import os
import sys
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: InsightFace not installed. Please run 'pip install -r requirements.txt' first.")
    sys.exit(1)

def setup_models():
    """Download and setup InsightFace models"""
    print("AI Surveillance Camera v1.0 - Model Setup")
    print("=" * 50)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Available models
    available_models = {
        'buffalo_s': 'Fast, good accuracy - recommended for real-time',
        'buffalo_m': 'Balanced speed/accuracy - recommended for most users', 
        'buffalo_l': 'Slow, best accuracy - for high-accuracy requirements'
    }
    
    print("\nAvailable InsightFace models:")
    for model, description in available_models.items():
        print(f"  {model}: {description}")
    
    # Get user choice
    print(f"\nCurrent default model: buffalo_m")
    choice = input("Enter model name to download (or press Enter for buffalo_m): ").strip()
    
    if not choice:
        choice = 'buffalo_m'
    
    if choice not in available_models:
        print(f"Error: Invalid model '{choice}'. Please choose from: {list(available_models.keys())}")
        return False
    
    print(f"\nDownloading {choice} model...")
    
    try:
        # Initialize FaceAnalysis - this will download the model
        app = FaceAnalysis(name=choice)
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print(f"✓ Successfully downloaded and verified {choice} model")
        print(f"✓ Model is ready for use")
        
        # Update the main script if user chose a different model
        if choice != 'buffalo_m':
            update_main_script(choice)
            
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        return False

def update_main_script(model_name):
    """Update the main script to use the selected model"""
    try:
        with open('test.py', 'r') as f:
            content = f.read()
        
        # Update MODEL_NAME
        content = content.replace(
            "MODEL_NAME = 'buffalo_m'",
            f"MODEL_NAME = '{model_name}'"
        )
        
        with open('test.py', 'w') as f:
            f.write(content)
        
        print(f"✓ Updated test.py to use {model_name} model")
        
    except Exception as e:
        print(f"Warning: Could not update test.py automatically: {e}")
        print(f"Please manually change MODEL_NAME to '{model_name}' in test.py")

def check_requirements():
    """Check if all requirements are installed"""
    required_packages = ['cv2', 'numpy', 'onnxruntime', 'insightface']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'onnxruntime':
                import onnxruntime
            elif package == 'insightface':
                import insightface
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Error: Missing required packages: {missing}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup function"""
    print("Checking requirements...")
    
    if not check_requirements():
        sys.exit(1)
    
    print("✓ All requirements satisfied")
    
    if not setup_models():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup complete! 🎉")
    print("\nTo run the AI Surveillance Camera:")
    print("  python test.py")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - The system will automatically detect your webcam")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
