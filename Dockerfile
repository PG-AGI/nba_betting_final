# Use a Python base image compatible with the project
FROM python:3.9

RUN apt update && apt-get upgrade -y
RUN apt install -y build-essential postgresql postgresql-contrib apache2

RUN pip install --upgrade pip
RUN pip install tensorflow autokeras psycopg2-binary matplotlib numpy pandas scrapy
RUN pip install datetime pytz sqlalchemy sqlalchemy sqlalchemy.orm scikit-base scikit-learn
RUN pip install pendulum pycaret scipy seaborn ydata_profiling requests
RUN pip install itemadapter nbconvert mlflow python-dotenv dash_bootstrap_components
RUN pip install apache-airflow apache-airflow-providers-common-sql apache-airflow-providers-ftp apache-airflow-providers-http apache-airflow-providers-imap apache-airflow-providers-sqlite

RUN pip install jupyter
RUN python -m ipykernel install --user --name=python3
RUN python -m ipykernel install --user --name=nba_kernal

# Command to configure apache server
RUN a2enmod proxy
RUN a2enmod proxy_http
RUN printf "<VirtualHost *:80>\n\tProxyPass / http://localhost:5000/\n\tProxyPassReverse / http://localhost:5000/\n</VirtualHost>\n" > /etc/apache2/sites-enabled/000-default.conf

EXPOSE 5000
EXPOSE 80

# Set up the working directory in the container
RUN mkdir -p /app
WORKDIR /app

# Copy the project files into the container
COPY . /app

RUN chmod +x /app/run.sh
CMD ["bash", "/app/run.sh"]