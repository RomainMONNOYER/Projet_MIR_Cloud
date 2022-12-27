FROM python:3.10

ENV PYTHONUNBUFFERED 1
EXPOSE 8000


COPY ./app /app
WORKDIR /app
#
##Download image database
#RUN wget https://github.com/sidimahmoudi/facenet_tf2/releases/download/AI_MIR_CLOUD/MIR_DATASETS_B.zip
#RUN unzip MIR_DATASETS_B.zip
#RUN mv MIR_DATASETS_B/ media/MIR_DATASETS_B
#
##Download ORB descriptor
#RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lDIBkJz4pl_vIwje9AAertIZLUdMOhy9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lDIBkJz4pl_vIwje9AAertIZLUdMOhy9" -O 'orb_data.txt' && rm -rf /tmp/cookies.txt
#RUN mv orb_data.txt media/ORB/data.txt
#
#
#RUN rm -r MIR_DATASETS_B.zip
COPY ./docker-dev/requirements.txt /tmp
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 manage.py makemigrations && python3 manage.py migrate && python3 manage.py loaddata features/features.json


#CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]
#Todo Fix number of worker to (2 x $num_cores) + 1 create a config file after
CMD [ "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2","Projet_MIR_Cloud.wsgi:application"]