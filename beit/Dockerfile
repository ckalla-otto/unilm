FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV LANG en_US.UTF-8

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install python3-dev build-essential curl unzip gnupg2 jq python3-pip libprotobuf-dev --yes --no-install-recommends

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# Installing gcloud package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh  
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN mkdir -p /beit-embedding-training
WORKDIR /beit-embedding-training
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN curl https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth --output /root/.cache/torch/hub/checkpoints/beit_base_patch16_224_pt22k_ft22k.pth
ENV PYTHONPATH /beit-embedding-training
COPY . .
RUN chmod +x copy_data_from_bucket_sampled.sh
RUN chmod +x copy_data_from_bucket.sh
RUN chmod +x run_training.sh
ENTRYPOINT ["/beit-embedding-training/run_training.sh"]
