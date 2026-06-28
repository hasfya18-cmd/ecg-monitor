FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy the requirements file first to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure data directory exists and has full write permissions for any user UID
RUN mkdir -p /app/data && chmod -R 777 /app /app/data

# Command to run the application using Gunicorn, wrapped in a shell to expand PORT variable
CMD ["sh", "-c", "gunicorn -w 1 --threads 8 -b 0.0.0.0:${PORT:-7860} app:app --timeout 120"]

