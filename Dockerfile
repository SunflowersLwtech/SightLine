FROM python:3.12-slim

# System dependencies: build tools for insightface Cython, OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to reduce image size (no longer needed at runtime)
RUN apt-get purge -y g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Download InsightFace buffalo_l models at build time
# (avoids downloading at runtime, which would slow cold starts)
RUN python -c "\
from insightface.app import FaceAnalysis; \
import os; \
app = FaceAnalysis(name='buffalo_l', \
    root=os.path.expanduser('~/.insightface'), \
    providers=['CPUExecutionProvider'], \
    allowed_modules=['detection', 'recognition']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace buffalo_l models downloaded successfully')"

# Copy backend source (excluding iOS, docs, dev, tests via .dockerignore)
COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
