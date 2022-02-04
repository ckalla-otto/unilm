#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR=${SCRIPT_DIR}/../../

# shellcheck disable=SC2164
cd "${ROOT_DIR}"

DATA_FOLDER="generated/cache/basic_color/"
mkdir $DATA_FOLDER
gsutil -m rsync -r gs://brain-ai-image-dev-artifacts/dali_usecase-basic-color/7bef03ae60b5a2278e702b67a81a04c630b3c348/tensorflow_datasets/dataset_builder_basic_color/0.10.0 $DATA_FOLDER
