# -*- coding: utf-8 -*-

from flask import Flask, jsonify
from flask import render_template
from flask import request

import json

# creates a Flask application, named app
app = Flask(__name__)

from diaparser.parsers import Parser

# parsers instances
parsers = {}

# a route where we will display a welcome message via an HTML template
@app.route("/", methods=['POST','GET'])
def parse():
    if request.method == 'POST':  # this block is only entered when the form is submitted
        sentences = request.form['sentences']
        language = request.form['language']
        if language in parsers:
            parser = parsers[language]
        else:
            parser = Parser.load('', lang=language)
            parsers[language] = parser
        parsed = parser.predict(sentences, text=language).sentences
        parsed_sentences = json.dumps([s.to_tokens() for s in parsed])

        return parsed_sentences

    data = {
        'parser_message': "Type sentences to parse.",
        'language': 'en',
        'sentences': ''
    }
    return render_template('index.html', **data)


@app.route('/elg', methods=['POST'])
def elg():
    content = request.json.get('content', None)
    params = request.json.get('params', {})
    language = params['language'] if 'language' in params else 'en'
    if language in parsers:
        parser = parsers[language]
    else:
        parser = Parser.load('', lang=language)
        parsers[language] = parser
    sentences = parser.predict(content, text=language).sentences
    parsed_sentences = [s.to_tokens() for s in sentences]

    # https://european-language-grid.readthedocs.io/en/release1.1.0/all/A2_API/LTInternalAPI.html#response-structure
    response = {
        'type': 'annotations',
        'annotations': {
            'start': 0,
            'end': len(content),
            'features': parsed_sentences
        }
    }

    return jsonify(response)
    

# run the application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
