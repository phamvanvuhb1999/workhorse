FROM python:3.7-stretch
# RUN apt-get update -y
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm requirements.txt
WORKDIR /app
COPY . .
