FROM python:3.10

ENV PYTHONUNBUFFERED 1
EXPOSE 8000


COPY ./app /app
WORKDIR /app
COPY ./docker-dev/requirements.txt /tmp
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 manage.py makemigrations && python3 manage.py migrate


#CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]
#Todo Fix number of worker to (2 x $num_cores) + 1 create a config file
CMD [ "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2","Projet_MIR_Cloud.wsgi:application"]