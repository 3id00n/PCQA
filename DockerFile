# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU-only build)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Default command: run your main script
CMD ["python", "src/n-jet_fitting.py"]
# CMD ["bash", "-c", "python src/n-jet_fitting.py && python src/pcqa_demo.py"]
