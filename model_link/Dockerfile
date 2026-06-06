FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN playwright install --with-deps chromium

COPY ./server /code/server

ENV PYTHONPATH=/code/server:/code

EXPOSE 8000

WORKDIR /code/server

CMD ["uvicorn", "mcp_server:streamable_http_app", "--host", "0.0.0.0", "--port", "8000"]
