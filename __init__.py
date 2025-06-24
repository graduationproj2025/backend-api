__version__ = "1.0.0"  # تصحيح: استخدام __version__ بدلاً من version_
__all__ = ['app', 'model_def']  # تصحيح: استخدام __all__ بدلاً من all_

# تهيئة تطبيق Flask
from flask import Flask
app = Flask(__name__)

# تحميل النموذج عند التشغيل
try:
    from .model_def import load_model  # إضافة النقطة للإشارة إلى الاستيراد النسبي
    model = load_model('model.pth')
except ImportError as e:
    print(f"Import Error: {e}")  # تصحيح: استخدام print بدلاً من printf
except Exception as e:
    print(f"Model Loading Error: {e}")  # تصحيح: استخدام f-string