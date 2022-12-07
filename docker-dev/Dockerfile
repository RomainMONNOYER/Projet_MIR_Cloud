FROM python:3.10

ENV PYTHONUNBUFFERED 1
EXPOSE 8000



COPY requirements.txt /tmp/app/
RUN apt-get update && apt-get upgrade -y
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/app/requirements.txt

WORKDIR /app





#CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]