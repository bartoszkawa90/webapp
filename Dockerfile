FROM python:3.9

ENV PYTHONUNBUFFERED 1

WORKDIR webapp

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD python manage.py runserver 0.0.0.0:8000