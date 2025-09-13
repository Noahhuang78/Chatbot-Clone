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
FROM python:3.10.10-slim

WORKDIR /app

# Install system dependencies for Chroma + sqlite3
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY flask-server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY flask-server/ ./

# Copy React build into Flask static folder
COPY --from=frontend-build /app/frontend/dist ./static
# If using Vite, change "dist" to your build folder name

# Expose port (Render uses $PORT)
ENV PORT=8080
EXPOSE $PORT

# Run Flask
CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "test_server:app"]
