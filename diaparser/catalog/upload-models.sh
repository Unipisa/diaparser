#!/usr/bin/env bash
#
# Upload models and catalog to GitHub
#
# upload-models.sh TOKEN [RELEASE] [MODEL_DIR]
#

if [ $# -eq 0 ]; then
    echo Usage: upload-models.sh TOKEN [RELEASE] [MODEL_DIR]
    exit
fi

owner=Unipisa
repo=diaparser
token=$1
tag=${2:-v1.0}
version=1

MODEL_DIR=${3:-../../exp}

# Define variables
GH_API="https://api.github.com"
GH_REPO="$GH_API/repos/$owner/$repo"
GH_TAGS="$GH_REPO/releases/tags/$tag"
AUTH="Authorization: token $token"
ACCEPT="Accept: application/vnd.github.v3+json"

upload_asset () {
    local filename=$1
    local name=$2
    local asset="https://uploads.github.com/repos/$owner/$repo/releases/$relid/assets?name=$name"
    curl --data-binary @"$filename" -sH "$AUTH" -H "Content-Type: application/octet-stream" $asset
}

# Get ID of the release
response=$(curl -sH "$AUTH" $GH_TAGS)
eval $(echo "$response" | grep -m 1 "id.:" | grep -w id | tr : = | tr -cd '[[:alnum:]]=')
[ "$id" ] || { echo "Error: Failed to get release id for tag: $tag"; echo "$response" | awk 'length($0)<100' >&2; exit 1; }
relid=$id

# Delete catalog
# Get ID of the catalog asset
name=catalog-${version}.json
eval $(curl -sH "$AUTH" $GH_REPO/releases/$relid/assets | grep -B2 '"name": "'$name'"' |head -1|tr : = | tr -d '" ,')
[ "$id" ] || { echo "Error: Failed to get id for asset: $name"; }

# Delete the old asset
curl -X DELETE -H "$AUTH" -H "$ACCEPT" $GH_REPO/releases/assets/$id

# upload new catalog
upload_asset ${MODEL_DIR}/$name $name
echo Uploaded $name

MODELS=(
    ar_padt.asafaya
    bg_btb.DeepPavlov
    ca_ancora.mbert
    cs_pdt_cac_fictree.DeepPavlov
    de_htd.dbmdz-bert-base
    en_ewt.electra-base
    en_ptb.electra-base
    es_ancora.mbert
    et_edt_ewt.mbert
    fi_tdt.TurkuNLP
    fr_sequoia.camembert-large
    it_isdt.dbmdz-electra-xxl
    ja_gsd.mbert
    la_ittb_llct.mbert
    lt_alksnis.mbert
    lv_lvtb.mbert
    no_nynorsk.mbert
    nl_alpino_lassysmall.wietsedv
    pl_pdb_lfg.dkleczek
    ro_rrt.mbert
    ru_syntagrus.DeepPavlov
    sk_snk.mbert
    sv_talbanken.KB
    ta_ttb.mbert
    uk_iu.TurkuNLP
    zh_ctb7.hfl-electra-base
)

# upload models
for model in ${MODELS[@]}; do
    echo "Uploading $model..."
    filename=$MODEL_DIR/$model/model
    upload_asset $filename $model
done

# ----------------------------------------------------------------------
# Manage assets
# List:
# curl -X GET -H 'Accept: application/vnd.github.v3+json' https://api.github.com/repos/Unipisa/diaparser/releases/$relid/assets
# Delete:
# curl -X DELETE -H 'Authorization: token $TOKEN' -H 'Accept: application/vnd.github.v3+json' https://api.github.com/repos/Unipisa/diaparser/releases/assets/27145683
