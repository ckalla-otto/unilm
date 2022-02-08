#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR=${SCRIPT_DIR}/../../

DATA_FOLDER="generated/cache/hackathon/"
mkdir -p $DATA_FOLDER
gsutil -m rsync -r gs://brain-ai-image-dev-artifacts/hackathon_2022-02/tensorflow_datasets/dataset_builder_hackathon/0.2.0 $DATA_FOLDER
