#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR=${SCRIPT_DIR}/../../

# shellcheck disable=SC2164
cd "${ROOT_DIR}"

DATA_FOLDER="generated/cache/basic_color_samples"
gsutil cp gs://brain-ai-image-dev-artifacts/dali_usecase-basic-color/7bef03ae60b5a2278e702b67a81a04c630b3c348/tensorflow_datasets/dataset_builder_basic_color/0.10.0/dataset_builder_basic_color-train.tfrecord-00006-of-00064 $DATA_FOLDER/dataset_builder_basic_color-train.tfrecord-00006-of-00064
gsutil cp gs://brain-ai-image-dev-artifacts/dali_usecase-basic-color/7bef03ae60b5a2278e702b67a81a04c630b3c348/tensorflow_datasets/dataset_builder_basic_color/0.10.0/dataset_builder_basic_color-validation.tfrecord-00000-of-00008 $DATA_FOLDER/dataset_builder_basic_color-validation.tfrecord-00000-of-00008

