__version__ = "1.0.0"
__all__ = ['app', 'model_def']

from flask import Flask
app = Flask(__name__)

try:
    from model_def import MultiImageClassifier  # استيراد مباشر بدون نقطة
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiImageClassifier().load_model("model.pth", device)
except Exception as e:
    print(f"Initialization Error: {e}")
    model = None