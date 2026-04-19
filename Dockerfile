# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Build with the backend URL pointing to the same container
RUN npm run build


# ── Stage 2: Python backend + serve frontend static files ────────────────────
FROM python:3.9-slim

WORKDIR /app

# System deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/   backend/
COPY src/        src/
COPY data/       data/

# Copy built frontend into a static folder the backend will serve
COPY --from=frontend-build /app/frontend/dist static/

# Copy env template (real secrets come in at runtime via env vars or .env mount)
COPY .env.example .env.example

EXPOSE 8000

# Serve frontend static files via FastAPI's StaticFiles + run the API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
