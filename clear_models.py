import shutil
from pathlib import Path

def clear_models():
    """Clear all downloaded models and caches"""
    print("🗑️ Clearing Models and Caches...")
    
    # Clear HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        try:
            shutil.rmtree(hf_cache)
            print("✅ HuggingFace cache cleared")
        except Exception as e:
            print(f"❌ Error clearing HuggingFace cache: {e}")
    else:
        print("ℹ️ HuggingFace cache not found")
    
    # Clear local cache
    local_cache = Path("cache")
    if local_cache.exists():
        try:
            shutil.rmtree(local_cache)
            print("✅ Local cache cleared")
        except Exception as e:
            print(f"❌ Error clearing local cache: {e}")
    else:
        print("ℹ️ Local cache not found")
    
    # Clear Backend cache
    backend_cache = Path("Backend/cache")
    if backend_cache.exists():
        try:
            shutil.rmtree(backend_cache)
            print("✅ Backend cache cleared")
        except Exception as e:
            print(f"❌ Error clearing backend cache: {e}")
    else:
        print("ℹ️ Backend cache not found")
    
    print("\n🎉 All models and caches cleared successfully!")
    print("💡 You can now run app.py again")

if __name__ == "__main__":
    clear_models() 