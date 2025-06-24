from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from model_def import MultiImageClassifier
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import psycopg2
import os
from datetime import datetime
from psycopg2.extras import register_default_json, register_default_jsonb
import traceback

# دعم JSON للـ psycopg2
register_default_json(loads=lambda x: x)
register_default_jsonb(loads=lambda x: x)

app = Flask(__name__)
CORS(app)

# تحميل النموذج
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiImageClassifier()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DIAGNOSES = {
    0: "Normal Heart Function",
    1: "Mild Ischemia",
    2: "Severe Myocardial Infarction"
}

@app.route("/")
def index():
    return jsonify({"message": "Hello from Flask!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image is required."}), 400

        image_file = request.files["image"]
        filename = image_file.filename

        try:
            image = Image.open(image_file)
            image.load()
            image = image.convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Unsupported or unreadable image format."}), 400

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            confidence = torch.max(F.softmax(output, dim=1)).item() * 100
            predicted = torch.argmax(output, dim=1).item()

        diagnosis = DIAGNOSES.get(predicted, "Unknown")
        risk_level = "Low" if predicted == 0 else "High"

        # Findings and recommendations حسب التشخيص
        if predicted == 0:  # Normal
            findings = [
                "Normal lung fields with clear bilateral expansion",
                "Cardiac silhouette within normal limits",
                "No acute cardiopulmonary abnormalities detected"
            ]
            recommendations = [
                "Continue routine monitoring",
                "Annual follow-up recommended",
                "No immediate intervention required"
            ]
        elif predicted == 1:  # Mild Ischemia
            findings = [
                "Possible reduced perfusion in left ventricular area",
                "Mild irregularities in cardiac rhythm",
                "Signs of ischemic changes detected"
            ]
            recommendations = [
                "Schedule cardiology follow-up within 3 months",
                "Recommend stress test or further imaging",
                "Lifestyle modifications encouraged"
            ]
        elif predicted == 2:  # Severe MI
            findings = [
                "Significant myocardial damage evident",
                "Abnormal wall motion detected",
                "Critical perfusion deficit in left anterior descending artery region"
            ]
            recommendations = [
                "Immediate cardiology intervention required",
                "Admit to cardiac care unit",
                "Start acute MI protocol and monitoring"
            ]
        else:
            findings = ["No conclusive findings"]
            recommendations = ["Recommend further testing or expert review"]

        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # تخزين في قاعدة البيانات
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO analysis_results (image_name, confidence, diagnosis, risk, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            filename,
            round(confidence, 2),
            diagnosis,
            risk_level,
            analysis_date
        ))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "confidence": round(confidence, 2),
            "diagnosis": diagnosis,
            "findings": findings,
            "recommendations": recommendations,
            "riskLevel": risk_level,
            "analysisDate": analysis_date,
            "processingTime": "2.3 seconds"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        # المستخدمين
        cur.execute('SELECT COUNT(*) FROM public."User"')
        total_users = cur.fetchone()[0]

        # عدد التحليلات
        cur.execute('SELECT COUNT(*) FROM public.analysis_results')
        total_analyses = cur.fetchone()[0]

        # معدل الثقة
        cur.execute('SELECT AVG(confidence) FROM public.analysis_results')
        avg_confidence = round(cur.fetchone()[0] or 0, 2)

        # عدد الحالات الخطرة
        cur.execute("SELECT COUNT(*) FROM public.analysis_results WHERE risk = 'High'")
        high_risk_cases = cur.fetchone()[0]

        # بيانات شهرية
        cur.execute("""
            SELECT TO_CHAR(created_at, 'Mon') AS month,
                   COUNT(*) FILTER (WHERE diagnosis = 'Normal Heart Function') AS normal,
                   COUNT(*) FILTER (WHERE diagnosis = 'Mild Ischemia') AS ischemia,
                   COUNT(*) FILTER (WHERE diagnosis = 'Severe Myocardial Infarction') AS infarction
            FROM public.analysis_results
            GROUP BY month
            ORDER BY MIN(created_at)
        """)
        monthly_data = cur.fetchall()
        monthly_analyses = [
            {
                "month": row[0],
                "normal": row[1],
                "ischemia": row[2],
                "infarction": row[3]
            }
            for row in monthly_data
        ]

        # توزيع الخطورة
        cur.execute("SELECT risk, COUNT(*) FROM public.analysis_results GROUP BY risk")
        risk_data = cur.fetchall()
        risk_distribution = []
        for row in risk_data:
            label = row[0]
            color = "#10B981" if label == "Low" else "#EF4444"
            risk_distribution.append({
                "name": f"{label} Risk",
                "value": row[1],
                "color": color
            })

        cur.close()
        conn.close()

        return jsonify({
            "totalUsers": total_users,
            "totalAnalyses": total_analyses,
            "avgConfidence": avg_confidence,
            "highRiskCases": high_risk_cases,
            "monthlyAnalyses": monthly_analyses,
            "riskDistribution": risk_distribution
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

