# Use a Python base image compatible with the project
FROM python:3.10

# Set up the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install psycopg2-binary
RUN pip install sqlalchemy==1.4.22
RUN for i in {1..5}; do pip install --default-timeout=100 -r requirements.txt && break || sleep 15; done


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


