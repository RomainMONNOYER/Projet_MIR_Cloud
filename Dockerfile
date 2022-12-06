FROM python:3.10

ENV PYTHONUNBUFFERED 1
EXPOSE 8000

WORKDIR /app

COPY requirements.txt /app
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get upgrade -y





#CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]