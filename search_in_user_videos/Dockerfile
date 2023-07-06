FROM python:3.10.12-alpine3.18

COPY . /app

WORKDIR /app

RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python", "run.py" ]