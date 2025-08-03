# 🚀 Setup Environment cho RAG Chatbot

## 📋 Yêu cầu hệ thống

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11 (khuyến nghị 3.10)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Storage**: 10GB free space
- **GPU**: Không bắt buộc nhưng khuyến nghị (NVIDIA với 6GB+ VRAM)

### Recommended Setup
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3060+ hoặc equivalent
- **Storage**: SSD với 20GB+ free space

## 🐍 Cài đặt Python Environment

### 1. Cài đặt Python
```bash
# Download Python 3.10 từ python.org
# Hoặc sử dụng conda
conda create -n rag_chatbot python=3.10
conda activate rag_chatbot
```

### 2. Tạo Virtual Environment
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

## 📦 Cài đặt Dependencies

### 1. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 2. Cài đặt PyTorch (CPU/GPU)
```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Cài đặt Core Dependencies
```bash
cd Backend
pip install -r requirements.txt
```

### 4. Optional: Cài đặt GPU Optimizations
```bash
# Nếu có GPU NVIDIA
pip install xformers
pip install flash-attn --no-build-isolation
```

## 🔧 Cấu hình bổ sung

### 1. Tạo thư mục cache
```bash
mkdir cache
```

### 2. Cài đặt Git LFS (nếu cần)
```bash
# Windows
# Download từ https://git-lfs.github.com/

# Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Mac
brew install git-lfs
```

## 🚨 Troubleshooting

### Lỗi thường gặp:

#### 1. Out of Memory
```bash
# Giảm batch size trong code
# Hoặc tăng swap space
```

#### 2. CUDA Error
```bash
# Kiểm tra CUDA version
nvidia-smi
# Cài đặt đúng version PyTorch
```

#### 3. Model Download Slow
```bash
# Sử dụng mirror
export HF_ENDPOINT=https://hf-mirror.com
```

#### 4. Gradio không chạy
```bash
# Cài đặt lại gradio
pip uninstall gradio
pip install gradio>=4.25.0
```

## ✅ Kiểm tra cài đặt

### 1. Test Python Environment
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. Test GPU (nếu có)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Test Model Loading
```bash
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

## 🎯 Chạy ứng dụng

### 1. Kích hoạt environment
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Chạy RAG Chatbot
```bash
cd Backend
python app.py
```

### 3. Truy cập ứng dụng
Mở browser và truy cập: `http://localhost:7860`

## 📊 Performance Tips

### Tối ưu cho CPU:
- Sử dụng 4-bit quantization (đã có sẵn)
- Giảm max_seq_length nếu cần
- Tăng swap space

### Tối ưu cho GPU:
- Cài đặt xformers và flash-attn
- Sử dụng GPU memory efficiently
- Monitor GPU usage với nvidia-smi

## 🔄 Update Dependencies

### Cập nhật packages:
```bash
pip install --upgrade -r requirements.txt
```

### Cập nhật models:
```bash
# Xóa cache để download lại models
rm -rf cache/
rm -rf ~/.cache/huggingface/
``` 