FROM python:3.10-slim

# Set up user to run the app
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the requirements file first to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Change ownership of /app to the user
RUN chown -R user:user /app
USER user

# Command to run the application using Gunicorn, dynamically reading PORT env var
CMD gunicorn -w 1 --threads 8 -b 0.0.0.0:${PORT:-7860} app:app --timeout 120

