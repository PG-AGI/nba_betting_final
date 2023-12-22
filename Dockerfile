# Use a Python base image compatible with the project
FROM python:3.10

# Set up the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install psycopg2-binary
RUN pip install sqlalchemy==1.4.22
RUN pip install numpy
RUN pip install pandas
RUN pip install jupyter
RUN pip install nvidia-cublas-cu11
RUN pip install nvidia-cuda-cupti-cu11
RUN pip install nvidia-cuda-nvrtc-cu11
RUN pip install nvidia-cuda-runtime-cu11
RUN pip install nvidia-cudnn-cu11
RUN pip install nvidia-cufft-cu11
RUN pip install nvidia-curand-cu11
RUN pip install nvidia-cusolver-cu11
RUN pip install nvidia-cusparse-cu11
RUN pip install nvidia-nccl-cu11
RUN pip install nvidia-nvtx-cu11

# # Copy requirements.txt and install_requirements.sh
# COPY requirements.txt install_requirements.sh ./

# Make the install script executable and run it
RUN pip install -r requirements.txt

RUN pip install spacy && python -m spacy download en_core_web_sm

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to be executed
ENTRYPOINT ["/entrypoint.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1


# Command to run the application
CMD ["python", "startup.py"]


