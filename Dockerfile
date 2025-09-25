# --------------------------
# Stage 1: Build React frontend
# --------------------------
FROM node:20 AS frontend-build

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy all frontend files
COPY frontend/ ./

# Build React app
RUN npm run build

# --------------------------
# Stage 2: Setup Flask backend
# --------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Chroma + sqlite3
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev chromium chromium-driver

# Install Python dependencies
COPY flask-server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY flask-server/ ./

# Copy React build into Flask static folder
COPY --from=frontend-build /app/frontend/dist ./static
# If using Vite, change "dist" to your build folder name

#default PORT env variable
ENV PORT=8080
ENV REDIS_URL=redis://redis:6379/0

# Start Gunicorn using the PORT env variable
CMD ["sh", "-c", "python -m gunicorn -b 0.0.0.0:${PORT} test_server:app"]

