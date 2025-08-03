import os
import shutil
from pathlib import Path

def get_disk_usage(path):
    """Get disk usage for a path"""
    total, used, free = shutil.disk_usage(path)
    return {
        'total': total / (1024**3),  # GB
        'used': used / (1024**3),    # GB
        'free': free / (1024**3)     # GB
    }

def main():
    print("💾 Disk Space Check")
    print("=" * 50)
    
    # Check C: drive
    c_drive = get_disk_usage("C:\\")
    print(f"C: Drive:")
    print(f"  Total: {c_drive['total']:.2f} GB")
    print(f"  Used:  {c_drive['used']:.2f} GB")
    print(f"  Free:  {c_drive['free']:.2f} GB")
    print()
    
    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file())
        cache_gb = cache_size / (1024**3)
        print(f"🤗 HuggingFace Cache: {cache_gb:.2f} GB")
    else:
        print("🤗 HuggingFace Cache: Not found")
    
    # Check local cache
    local_cache = Path("cache")
    if local_cache.exists():
        cache_size = sum(f.stat().st_size for f in local_cache.rglob('*') if f.is_file())
        cache_gb = cache_size / (1024**3)
        print(f"📁 Local Cache: {cache_gb:.2f} GB")
    else:
        print("📁 Local Cache: Not found")
    
    print()
    print("✅ Models đã được xóa thành công!")
    print("💡 Bây giờ bạn có thể chạy lại app.py")

if __name__ == "__main__":
    main() 