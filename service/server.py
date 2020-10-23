#!/usr/bin/env python

from diaparser.parsers import Parser
from fastapi import FastAPI
from pydantic import BaseModel
from typing_extensions import Literal
from typing import Callable

APP_NAME = 'DiaParser Server'
APP_VERSION = '0.1'
DEBUG = True

def start_app_handler(app: FastAPI) -> Callable:
    def load_affinity_estimator() -> None:
        #config = yaml.safe_load(open('config.yaml'))
        app.state.parser = Parser.load('en_ewt.electra-base')
    return load_affinity_estimator


def get_app() -> FastAPI:
    fast_app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=DEBUG)
    fast_app.add_event_handler("startup", start_app_handler(fast_app))
    return fast_app

app = get_app()

LANGUAGES = ('en', )

class InputText(BaseModel):
    text: str
    language: Literal[LANGUAGES]

@app.post("/parse")
async def predict(input_text: InputText):
    resp = {'parsed': [], 'text': input_text.text}
    dataset = app.state.parser.predict(input_text.text, text=input_text.language) 
    for s in dataset.sentences:
        resp['parsed'].append([el for el in zip(*s.values)])
    return resp
