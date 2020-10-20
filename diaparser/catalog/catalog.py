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
UPLOAD_URL = f'https://uploads.github.com/repos/Unipisa/diaparser/releases/{RELEASE}/assets'
UPLOAD_COMMAND = f'curl -X POST -H "Content-Type: application/zip" {UPLOAD_URL}'

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

    # Download catalog.json to obtain latest packages.
    logger.debug('Downloading catalog file...')
    # make request
    catalog_path = os.path.join(dir, 'catalog.json')
    request_file(
        f'{catalog_url}/catalog-{catalog_version}.json',
        catalog_path)
    # unpack results
    try:
        models = json.load(open(catalog_path))
    except:
        raise Exception(
            f'Cannot load model list. Please check your network connection, '
            f'or provided resource url and resource version.'
        )
    if lang not in models:
        raise Exception(f'Unsupported language: {lang}.')
    corpora = models[lang]
    if 'alias' in corpora:
        alias = corpora['alias']
        logger.info(f'"{alias}" is an alias for "{lang}"')
        corpora = models[alias]
    if corpus in corpora:
        model = corpora[corpus].get('model', None)
        if model:
            return  f'{catalog_url}/model'
    if 'default' in corpora:
        logger.info(f'Using {corpora["default"]} as corpus for {lang}')
        corpus = lang["default"]
        model = corpora[corpus].get('model', None)
        if model:
            return  f'{catalog_url}/model'
    return None


def upload(path, owner='Unipisa', repo='diaparser', version='v1.0', token=''):
    name = os.path.basename(path)
    GH_ASSET=f"https://uploads.github.com/repos/{owner}/{repo}/releases/{version}/assets?name={name}"
    curl = f'curl --data-binary @"{path}" -H "Authorization: token {token}" -H "Content-Type: application/zip" {GH_ASSET}'
    subprocess.run(curl.split())
