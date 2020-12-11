
from python:3.8

RUN pip install -r requirements.txt
RUN pip install flask
RUN pip install diaparser

COPY elg/index.html .

EXPOSE 8080
ENTRYPOINT python elg/service.py
