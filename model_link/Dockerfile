FROM python:3.15.0b1-alpine3.23 

WORKDIR /code 

COPY ./requirements.txt   /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

COPY ./server   /code/server

CMD ["python3", "-m", "mcp_server"]