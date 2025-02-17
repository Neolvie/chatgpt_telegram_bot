FROM python:3.8-slim

ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PYTHONDONTWRITEBYTECODE 1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-dev build-essential python3-venv ffmpeg

RUN mkdir -p /code
WORKDIR /code
ADD ./requirements.txt /code/requirements.txt
RUN pip3 install -r requirements.txt
ADD . /code

CMD ["bash"]
