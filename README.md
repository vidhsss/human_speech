# Multimodal Speech Recognition (Human-Inspired)

This project implements a human-inspired, robust multimodal speech recognition pipeline that integrates both audio (wav2vec2/HuBERT) and visual (ViT-based lip reading) cues. It features cross-modal attention fusion, mouth region-of-interest (ROI) extraction, and advanced transformer-based decoding, inspired by cognitive models of human perception under noisy conditions.

## Features
- **Multimodal Fusion:** Integrates audio and video (lip reading) for robust speech recognition.
- **Cross-Modal Attention:** Dynamically attends to audio and visual cues, mimicking human perception.
- **Mouth ROI Extraction:** Extracts mouth region from video frames for improved visual speech understanding.
- **CTC Loss & Transformer Decoder:** Supports CTC-based training and advanced transformer-based decoding.
- **Evaluation Script:** Easily evaluate and compare predictions to ground truth.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd multimodal_speech
   ```
2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio transformers timm opencv-python pandas tqdm dlib
   ```
3. **Download dlib face landmark model:**
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

## Usage

### **Training**
```bash
python train_multimodal.py
```

### **Evaluation/Inference**
```bash
python evaluate.py
```

### **Mouth ROI Extraction**
```python
from mouth_roi import extract_mouth_roi
frames = extract_mouth_roi('path/to/video.mp4', 'shape_predictor_68_face_landmarks.dat')
```

### **Advanced Fusion/Decoder**
- See `advanced_fusion.py` for a transformer-based decoder module you can plug into your model for more context-aware decoding.

## Citation & Acknowledgments
- Uses pretrained models from [HuggingFace Transformers](https://huggingface.co/transformers/)
- Mouth ROI extraction uses [dlib](http://dlib.net/)
- Inspired by [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [LipNet](https://github.com/mpc001/LipNet), and cognitive models of human speech perception.

## License
[Add your license here] 