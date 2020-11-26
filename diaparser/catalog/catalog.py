"""
Query the DiaParser model catalog.
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
import json
import hashlib
import logging
import subprocess

from .. import __version__, __models_version__

logger = logging.getLogger('diaparser')

RELEASE = 'v1.0'
DOWNLOAD_URL = f'https://github.com/Unipisa/diaparser/releases/download/{RELEASE}'

DEFAULT_CATALOG_URL = DOWNLOAD_URL
DEFAULT_CATALOG_VERSION = __models_version__

# set home dir for default
HOME_DIR = str(Path.home())
CACHE_DIR = os.path.join(HOME_DIR, '.cache/diaparser')


def get_md5(path):
    """
    Get the MD5 value of a path.
    """
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()


def file_exists(path, md5):
    """
    Check if the file at `path` exists and match the provided md5 value.
    """
    return os.path.exists(path) and get_md5(path) == md5


def download_file(url, path):
    """
    Download a URL into a file as specified by `path`.
    """
    verbose = logger.level in [0, 10, 20]
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))


def request_file(url, path, md5=None):
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.
    Args:
        url: of file to retrieve.
        path: where to store result.
        md5: the ecpected md5 of the file.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if file_exists(path, md5):
        logger.info(f'File exists: {path}.')
        return
    download_file(url, path)
    assert(not md5 or file_exists(path, md5))


def url_ok(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    return requests.head(url).ok


def available_models(lang, dir, catalog_url, catalog_version):
    """Download list of available models for :param lang:."""

    logger.debug('Downloading catalog file...')
    # make request
    catalog_path = os.path.join(dir, 'catalog.json')
    request_file(
        f'{catalog_url}/catalog-{catalog_version}.json',
        catalog_path)
    # unpack results
    try:
        catalog = json.load(open(catalog_path))
    except:
        raise Exception(
            'Cannot load model catalog. Please check your network connection, '
            'or provided resource URL and resource version.'
        )
    if lang not in catalog:
        raise Exception(f'Unsupported language: {lang}.')
    models = catalog[lang]
    if 'alias' in models:
        alias = models['alias']
        logger.info(f'"{alias}" is an alias for "{lang}"')
        corpora = models[alias]
    return models


def select(name=None, lang='en', corpus=None, bert=None,
           dir=CACHE_DIR,
           verbose=None,
           catalog_url=DEFAULT_CATALOG_URL,
           catalog_version=DEFAULT_CATALOG_VERSION,
           **kwargs):
    """
    Determines which model to download.
    If `name` is provided, the model with that name is returned.
    If just `lang` is specifiled, it selects the default model for the languagae.
    If `lang` and `corpus` are specified, it returns the model for the given language/corpus pair.
    Args:
        name (str): model name.
        lang (str): the language of the model.
        corpus (str): the corpus of the model.
    Returns:
        the URL from where to download the model.
    """

    # set global logging level
    logging_level = 'INFO' if verbose else 'ERROR'
    logger.setLevel(logging_level)

    # Check for a model with the given name.
    url = f'{catalog_url}/{name}'
    if url_ok(url):
        return url

    models = available_models(lang, dir, catalog_url, catalog_version)
    
    if corpus not in models:
        logger.info(f'Using {models["default"]} as corpus for {lang}')
        corpus = models.get("default", None)
    model = models[corpus].get('parse', None) if corpus in models else None
    if model:
        return f'{catalog_url}/{model}'
    return None


def available_processors(lang='en', corpus=None,
                         dir=CACHE_DIR,
                         verbose=None,
                         catalog_url=DEFAULT_CATALOG_URL,
                         catalog_version=DEFAULT_CATALOG_VERSION,
                         **kwargs):
    """
    Determines which processors (parse, tokenize, mwt) are available.
    If just `lang` is specifiled, it selects the default corpus for the languagae.
    If `lang` and `corpus` are specified, it returns the model for the given language/corpus pair.
    Args:
        lang (str): the language of the model.
        corpus (str): the corpus for the chosen language.
    Returns:
        dict of {'tokenize': tok_url, 'mwt': mwt_url} from where to download the preprocessors models.
    """
    models = available_models(lang, dir, catalog_url, catalog_version)
    if corpus not in models:
        logger.info(f'Using {models["default"]} as corpus for {lang}')
        corpus = models.get('default', None)
    return models.get(corpus, {})


def download_processors(lang, processors, dir,
                        download_url=DOWNLOAD_URL):
    paths = {}
    for proc,model in processors.items():
        try:
            path = os.path.join(dir, lang, proc, f'{model}.pt')
            request_file(f'{download_url}/{model}.pt', path)
            paths[proc + '_model_path'] = path
        except KeyError as e:
                raise Exception(
                    f'Cannot find the following processor and model name combination: '
                    f'{proc}, {model}. Please check if you have provided the correct model name.'
                ) from e
    return paths


def upload(path, owner='Unipisa', repo='diaparser', version='v1.0', token=''):
    name = os.path.basename(path)
    GH_ASSET = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{version}/assets?name={name}"
    curl = f'curl --data-binary @"{path}" -H "Authorization: token {token}" -H "Content-Type: application/zip" {GH_ASSET}'
    subprocess.run(curl.split())
