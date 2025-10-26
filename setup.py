#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: InsightFace not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

def setup_models(auto_yes=False):
    print("=" * 70)
    print("AI Surveillance Camera - Model Setup")
    print("=" * 70)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    models = {
        'buffalo_s': 'Speed',
        'buffalo_l': 'Accuracy'
    }
    
    existing = []
    for name in models.keys():
        model_path = models_dir / name
        if model_path.exists() and any(model_path.glob("*.onnx")):
            existing.append(name)
    
    if len(existing) == 2:
        print("\nBoth models already exist:")
        for name in models.keys():
            print(f"  * {name} - {models[name]}")
        print("\nSetup complete!")
        return True
    
    print(f"\nDownloading {2 - len(existing)} models to 'models/' folder:")
    for name, desc in models.items():
        if name not in existing:
            print(f"  * {name} - {desc}")
    
    print(f"\nSize: ~{(2 - len(existing)) * 60} MB")
    print("=" * 70)
    
    if not auto_yes:
        try:
            proceed = input("\nContinue? (y/n): ").strip().lower()
            if proceed not in ['y', 'yes']:
                return False
        except EOFError:
            print("\nAuto-continuing (non-interactive mode)...")
    else:
        print("\nAuto-continuing (--yes flag)...")
    
    print()
    success = 0
    
    for i, (name, desc) in enumerate(models.items(), 1):
        print(f"[{i}/2] {name}...", end=" ")
        
        try:
            model_path = models_dir / name
            if model_path.exists() and any(model_path.glob("*.onnx")):
                print("exists")
                success += 1
                continue
            
            # Download to temp location, then move to correct location
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                app = FaceAnalysis(name=name, root=tmpdir)
                app.prepare(ctx_id=0, det_size=(640, 640))
                
                # Move from temp to models folder
                src = Path(tmpdir) / 'models' / name
                if src.exists():
                    shutil.move(str(src), str(model_path))
                    print("done")
                    success += 1
                else:
                    print("error: model not found after download")
            
            # Clean up any zip files
            for zip_file in models_dir.glob("*.zip"):
                zip_file.unlink()
            
        except Exception as e:
            print(f"error: {e}")
    
    print("\n" + "=" * 70)
    if success == 2:
        print("Both models ready!")
        return True
    elif success > 0:
        print(f"{success}/2 models available")
        return True
    else:
        print("Failed to download models")
        return False

def check_requirements():
    required = ['cv2', 'numpy', 'onnxruntime', 'insightface', 'PyQt5']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'cv2':
                import cv2
            elif pkg == 'numpy':
                import numpy
            elif pkg == 'onnxruntime':
                import onnxruntime
                print(f"ONNX Runtime version: {onnxruntime.__version__}")
                providers = onnxruntime.get_available_providers()
                if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
                    print("GPU acceleration: Available")
                else:
                    print("GPU acceleration: Not available (CPU only)")
            elif pkg == 'insightface':
                import insightface
            elif pkg == 'PyQt5':
                import PyQt5
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def main():
    import sys
    auto_yes = '--yes' in sys.argv or '-y' in sys.argv
    
    if not check_requirements():
        sys.exit(1)
    
    print("Requirements OK\n")
    
    if not setup_models(auto_yes):
        sys.exit(1)
    
    print("\nSetup Complete!")
    print("Run: python gui_app_qt.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
