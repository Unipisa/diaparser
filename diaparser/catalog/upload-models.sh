#!/usr/bin/env bash
#
# Upload models and catalog to GitHub
#

OWNER=Unipisa
REPO=diaparser
TAG=LATEST
TOKEN=$1
VERSION=1

MODEL_DIR=../../exp

# upload catalog
./upload-github-release-asset.sh github_api_token=$TOKEN owner=$OWNER repo=$REPO tag=$TAG filename=${MODEL_DIR}/catalog-$VERSION.json

MODELS=(
    ar_padt.asafaya
    bg_btb.DeepPavlov
    cs_PDT.DeepPavlov
    nl_alpino.wietsedv
    en_ewt.electra-base
    et_edt_ewt.TurkuNLP
    fi_tdt.TurkuNLP
    fr_sequoia.camembert-large
    it_isdt.dbmdz-xxl
    lt_alksnis.wietsedve
    pl_pdb_pud.dkleczek
    ru_syntagrus.DeepPavlov
    sk_snk.KB
    ta_ttb.mbert
    uk_iu.TurkuNLP
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
