# tgi-validator


## Setup

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Test simple deployment

```bash
export HF_TOKEN=<your-huggingface-token>
export TGI_IMAGE_URI="ghcr.io/huggingface/text-generation-inference:2.4.1"

docker pull $TGI_IMAGE_URI

docker run --gpus all --shm-size 1g -e HF_TOKEN=$HF_TOKEN -p 8080:80 -v $PWD/data:/data $TGI_IMAGE_URI --model-id meta-llama/Llama-3.2-1B-Instruct
```

## Run validation tests
```bash

python validator.py \
--model-ids "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mixtral-8x22B-Instruct-v0.1" "meta-llama/Llama-3.2-11B-Vision-Instruct" "meta-llama/Llama-3.3-70B-Instruct" \
--token-values 1024 2048 4096 8192 16384 32768 \
--gpu-type nvidia-a10g \
--image-uri $TGI_IMAGE_URI
```


