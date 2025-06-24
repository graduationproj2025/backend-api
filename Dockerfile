# استخدم صورة بايثون الرسمية
FROM python:3.10-slim

# إعداد مجلد العمل
WORKDIR /app

# نسخ الملفات إلى الحاوية
COPY . /app

# تثبيت pip وتثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# فتح المنفذ
EXPOSE 10000

# أمر التشغيل
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
