FROM python:3.9-slim-bullseye

COPY . /app

WORKDIR /app

RUN python3 -m venv /opt/venv

RUN . /opt/venv/bin/activate && pip install --upgrade pip
RUN . /opt/venv/bin/activate && pip install -r requirements.txt

EXPOSE 5001

ENTRYPOINT . /opt/venv/bin/activate && exec python myapp.py