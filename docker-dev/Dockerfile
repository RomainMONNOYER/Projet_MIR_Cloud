FROM python:3.10

ENV PYTHONUNBUFFERED 1
EXPOSE 8000

WORKDIR /app

COPY ../ /app
RUN apt-get update && apt-get upgrade -y
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 manage.py makemigrations && python3 manage.py migrate


CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]