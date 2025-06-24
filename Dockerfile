# استخدمي صورة بايثون مناسبة
FROM python:3.10-slim

# ضبط مجلد العمل
WORKDIR /app

# نسخ الملفات
COPY . .

# تثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# تشغيل التطبيق عبر gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
