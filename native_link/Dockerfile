FROM python:3.12-slim 

WORKDIR /code

COPY ./requirements.txt  /code/requirements.txt

RUN  pip install --no-cache-dir --upgrade -r /code/requirements.txt 

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./client /code/client
 

ENV PYTHONPATH=/code/client:/code

ENV TERM=xterm-256color

WORKDIR /code/client

CMD ["python3", "tui.py"]
