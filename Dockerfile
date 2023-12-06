# FROM python:3.9

# ENV DockerHome=/Users/bartoszkawa/Desktop/REPOS/GitHub/Webappp/webapp

# WORKDIR $DockerHome

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED=1

# COPY . $DockerHOME

# # Install libgl1-mesa-glx
# RUN apt-get update && apt-get install -y libgl1-mesa-glx

# RUN pip3 install -r requirements.txt
# RUN pip install requests

# EXPOSE 8000

# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

FROM python:3.9

ENV PYTHONUNBUFFERED 1

WORKDIR webapp

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

COPY . .

CMD python manage.py runserver 0.0.0.0:80