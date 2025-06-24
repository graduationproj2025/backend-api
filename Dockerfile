# استخدام صورة أساسية خفيفة الوزن مع Python 3.9 (متوافق مع Render)
FROM python:3.9-slim

# تعطيل التحقق من إصدار pip لتجنب التحذيرات
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# إنشاء دليل العمل وتحديث النظام
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# نسخ متطلبات المشروع أولاً للاستفادة من طبقات Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# حل بديل لمشكلة __version__ إذا لم يكن هناك setup.py
RUN if [ ! -f /app/backend-api/__init__.py ]; then \
    echo "__version__ = '1.0.0'" > /app/backend-api/__init__.py; \
    fi

# تعيين منفذ التطبيق (يجب أن يتطابق مع Render)
EXPOSE 10000

# تشغيل التطبيق باستخدام Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "4", "--timeout", "120"]