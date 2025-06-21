from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from model_def import MultiImageClassifier
import torchvision.transforms as transforms
from PIL import Image

# تحديد الجهاز (GPU أو CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# عدد الصور المطلوبة (صورتين)
NUM_IMAGE_TYPES = 2

# تحميل النموذج (مدرب على صورتين)
model = MultiImageClassifier(num_classes=3, num_image_types=NUM_IMAGE_TYPES)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# أسماء الفئات
class_labels = ['Normal', 'Ischemia', 'Infarction']

# التحويلات على الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app)


@app.route('/api/hello')
def hello():
    return jsonify({'message': 'Hello from Flask!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام الصور
        files = request.files.getlist('images')

        # التحقق من عدد الصور
        if len(files) != NUM_IMAGE_TYPES:
            return jsonify({'error': f'Expected {NUM_IMAGE_TYPES} images.'}), 400

        # تحويل الصور إلى tensors
        tensors = []
        for file in files:
            img = Image.open(file.stream).convert('RGB')
            tensor = transform(img)
            tensors.append(tensor)

        # دمجهم في Tensor نهائي مناسب للنموذج
        input_tensor = torch.stack(tensors).unsqueeze(0).to(device)  # shape: (1, 2, 3, 224, 224)

        # التنبؤ
        with torch.no_grad():
            output, _ = model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(output.argmax(dim=1))

        # إرجاع النتيجة
        return jsonify({
            'class': class_labels[prediction],
            'confidence': float(probs[prediction])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

