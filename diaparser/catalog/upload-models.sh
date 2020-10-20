#!/usr/bin/env bash
#
# Upload models and catalog to GitHub
#

OWNER=Unipisa
REPO=diaparser
TAG=LATEST
TOKEN=3b26cf8754b46fd5f66d94618bce901e3e3309d6
VERSION=1

MODEL_DIR=../../exp

# upload catalog
./upload-github-release-asset.sh github_api_token=$TOKEN owner=$OWNER repo=$REPO tag=$TAG filename=${MODEL_DIR}/catalog-$VERSION.json

MODELS=(
    it_isdt.dbmdz-xxl
    en_ewt.electra-base
)

# upload models
for model in ${MODELS[@]}; do \
    ./upload-github-release-asset.sh github_api_token=$TOKEN owner=$OWNER repo=$REPO tag=$TAG filename=${MODEL_DIR}/$model/model name=$model;
done

# ----------------------------------------------------------------------
# Manage assets
# Find release ID
ID=32726187
# Upload:
# curl --data-binary @exp/it_isdt.dbmdz-xxl/model -H 'Authorization: token $TOKEN' -H 'Content-Type: application/octet-stream' https://uploads.github.com/repos/Unipisa/diaparser/releases/32726187/assets?name=it_isdt.dbmdz-xxl
# List:
# curl -X GET -H 'Accept: application/vnd.github.v3+json' https://api.github.com/repos/Unipisa/diaparser/releases/32726187/assets
# Delete:
# curl -X DELETE -H 'Authorization: token $TOKEN' -H 'Accept: application/vnd.github.v3+json' https://api.github.com/repos/Unipisa/diaparser/releases/assets/27145683
