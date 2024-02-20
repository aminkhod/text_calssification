FROM python:3.8-slim-bullseye
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
ENV FLASK_APP app
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
EXPOSE 5000
WORKDIR /app
RUN pip3 install -r requirements.txt
# RUN pip install tensorflow
ENTRYPOINT [ "python3", "api.py"  ]




COPY . .
CMD ["flask", "run"]
