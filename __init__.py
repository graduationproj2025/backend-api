__version__ = "1.0.0"  # الإصدار الأساسي
__all__ = ['app', 'model_def']  # تصدير الوحدات الرئيسية

# تهيئة تطبيق Flask
from flask import Flask
app = Flask(__name__)

# تحميل النموذج عند التشغيل
try:
    from model_def import load_model
    model = load_model('model.pth')
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Model Loading Error: {e}")