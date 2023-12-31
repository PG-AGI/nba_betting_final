# Use a Python base image compatible with the project
FROM python:3.10

# Set up the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies from requirements.txt
# RUN nvidia-cublas-cu11
# RUN nvidia-cuda-cupti-cu11
# RUN nvidia-cuda-nvrtc-cu11
# RUN nvidia-cuda-runtime-cu11
# RUN nvidia-cudnn-cu11
# RUN nvidia-cufft-cu11
# RUN nvidia-curand-cu11
# RUN nvidia-cusolver-cu11
# RUN nvidia-cusparse-cu11
# RUN nvidia-nccl-cu11
# RUN nvidia-nvtx-cu11
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
RUN pip install mlflow

RUN pip install jupyter
RUN python -m ipykernel install --user --name=python3
RUN python -m ipykernel install --user --name=nba_kernal

# # Copy requirements.txt and install_requirements.sh
# COPY requirements.txt install_requirements.sh ./

# Make the install script executable and run it
# RUN pip install -r requirements.txt

# RUN pip install spacy && python -m spacy download en_core_web_sm

# Copy the entrypoint script
# COPY entrypoint.sh /entrypoint.sh

# # Make the entrypoint script executable
# RUN chmod +x /entrypoint.sh

# # Set the entrypoint script to be executed
# ENTRYPOINT ["/entrypoint.sh"]

# # Health check


# RUN python startup.py
# RUN python notebooks/AutoDL_Cls.py
# RUN python notebooks/AutoDL_Reg.py
# RUN python notebooks/AutoML_Cls.py
# RUN python notebooks/AutoML_Reg.py
# RUN python notebooks/Exploratory_Analysis.py

# Command to run the application
CMD ["python", "startup.py"]


