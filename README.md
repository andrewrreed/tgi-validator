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

python validator.py --model-id "meta-llama/Llama-3.2-11B-Vision-Instruct" --token-values 1024 2048 4096 8192 16384 --gpu-type "nvidia-a10g"
```

