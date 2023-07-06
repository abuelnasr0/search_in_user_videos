FROM python:3.10.12-alpine3.18

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python", "run.py" ]