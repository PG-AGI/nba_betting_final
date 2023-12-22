# Use a Python base image compatible with the project
FROM python:3.10

# Set up the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install psycopg2-binary
RUN pip install datetime
RUN pip install numpy
RUN pip install pandas
RUN pip install pytz
RUN pip install sqlalchemy
RUN pip install scrapy
RUN pip install sqlalchemy
RUN pip install sqlalchemy.orm
RUN pip install scikit-base
RUN pip install scikit-learn
RUN pip install pendulum
RUN pip install apache-airflow
RUN pip install apache-airflow-providers-common-sql
RUN pip install apache-airflow-providers-ftp
RUN pip install apache-airflow-providers-http
RUN pip install apache-airflow-providers-imap
RUN pip install apache-airflow-providers-sqlite
RUN pip install tensorflow
RUN pip install autokeras
RUN pip install pycaret
RUN pip install scipy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install ydata_profiling
RUN pip install requests
RUN pip install itemadapter
RUN pip install nbconvert
# # Copy requirements.txt and install_requirements.sh
# COPY requirements.txt install_requirements.sh ./

# Make the install script executable and run it
# RUN pip install -r requirements.txt

# RUN pip install spacy && python -m spacy download en_core_web_sm

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


