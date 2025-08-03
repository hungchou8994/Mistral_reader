# 🔓 PDF Troubleshooting Guide

## ❌ Lỗi "PDF is encrypted/password protected"

### 🔍 Nguyên nhân:
- PDF có password bảo vệ
- PDF được mã hóa (encrypted)
- PDF có DRM (Digital Rights Management)

### 🛠️ Cách khắc phục:

#### 1. **Xóa password protection**
```bash
# Sử dụng Adobe Acrobat
1. Mở PDF trong Adobe Acrobat
2. File → Properties → Security
3. Security Method → No Security
4. Save file
```

#### 2. **Sử dụng online tools**
- [SmallPDF](https://smallpdf.com/unlock-pdf)
- [ILovePDF](https://www.ilovepdf.com/unlock_pdf)
- [PDF24](https://tools.pdf24.org/en/unlock-pdf)

#### 3. **Print to PDF method**
```
1. Mở PDF trong browser hoặc PDF reader
2. Ctrl+P (Print)
3. Chọn "Save as PDF"
4. Lưu file mới (sẽ không có password)
```

#### 4. **Sử dụng command line (Linux/Mac)**
```bash
# Cài đặt qpdf
sudo apt-get install qpdf

# Xóa password
qpdf --password=YOUR_PASSWORD input.pdf output.pdf
```

### 🧪 Kiểm tra PDF trước khi upload:

```python
# Chạy script kiểm tra
python pdf_checker.py
```

### 📋 Checklist trước khi upload:

- [ ] PDF không có password
- [ ] PDF có thể extract text (không phải image-only)
- [ ] File size hợp lý (< 50MB)
- [ ] PDF có nội dung text (không phải scan)

### 🚨 Các loại PDF không hỗ trợ:

1. **Image-based PDF**: Chỉ chứa hình ảnh, không có text
2. **Scanned PDF**: PDF scan từ giấy
3. **DRM Protected**: Có bảo vệ bản quyền
4. **Corrupted PDF**: File bị hỏng

### 💡 Tips:

- **OCR**: Nếu PDF là scan, sử dụng OCR tools trước
- **Convert**: Chuyển đổi sang text-based PDF
- **Compress**: Giảm kích thước file nếu quá lớn
- **Test**: Kiểm tra PDF trước khi upload

### 🔧 Tools hữu ích:

- **Adobe Acrobat**: Xử lý PDF chuyên nghiệp
- **Google Drive**: Upload và convert PDF
- **Microsoft Word**: Mở và save lại PDF
- **Online OCR**: Chuyển scan thành text

### 📞 Support:

Nếu vẫn gặp vấn đề, hãy:
1. Kiểm tra PDF với `pdf_checker.py`
2. Thử các phương pháp trên
3. Upload PDF mới đã được xử lý 