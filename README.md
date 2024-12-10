# tgi-validator


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export HF_TOKEN=<your-huggingface-token>

python validator.py \
    --model-ids "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --token-values 2048 4096 8192 16384 \
    --gpu-type nvidia-a10g \
    --image-uri "ghcr.io/huggingface/text-generation-inference:2.4.1"
```

