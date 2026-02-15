---
description: How to train the Looped Transformer on AWS GPU instances
---

# AWS Training Workflow for Looped Transformer (546M params)

## Step 1: Launch AWS Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Settings:
   - **Name**: `nanochat-looped-training`
   - **AMI**: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)` 
     - Search "Deep Learning AMI GPU" in the AMI marketplace
     - This comes with NVIDIA drivers, CUDA, and cuDNN pre-installed
   - **Instance type**: `p3.16xlarge` (8× V100 32GB, ~$24/hr) or `p4d.24xlarge` (8× A100 40GB, ~$32/hr)
     - For budget: `g5.4xlarge` (1× A10G 24GB, ~$1.62/hr) — slower but cheaper for testing
   - **Key pair**: Select or create a key pair (e.g., `nanochat-key`)
   - **Storage**: 200 GB gp3 (for model checkpoints + data)
   - **Security group**: Allow SSH (port 22) from your IP

3. Click **Launch Instance**

4. Note your instance's **Public IPv4 address**

## Step 2: Connect to Instance

```bash
# From your local machine (replace with your key and IP)
ssh -i "nanochat-key.pem" ubuntu@<YOUR_INSTANCE_IP>
```

## Step 3: Setup Environment (~5 min)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and pip (should already be there on Deep Learning AMI)
python3 --version  # should be 3.10+

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone the repo
git clone https://github.com/Pandurangmopgar/nanochat.git
cd nanochat

# Install dependencies
uv sync
```

## Step 4: Verify GPU Setup (~1 min)

```bash
# Check GPUs are visible
nvidia-smi

# Quick Python GPU check
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"
```

## Step 5: Quick Sanity Check (~2 min)

Run 10 training steps to verify everything works:

```bash
uv run python -m scripts.looped_train \
  --depth=36 \
  --num-iterations=10 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --device-batch-size=8
```

**What to check:**
- Model config prints correctly (546M params, P-R-C = 2,3,1)
- Loss prints for each step (should be ~10-11 at start for vocab=32768)
- No OOM errors — if OOM, reduce `--device-batch-size` to 4 or 2
- No NaN losses

## Step 6: Launch Full Training

### Single GPU (g5.4xlarge / single V100):
```bash
uv run python -m scripts.looped_train \
  --run=looped_500m_v1 \
  --depth=36 \
  --save-every=1000 \
  --eval-every=250 \
  --sample-every=2000 \
  --device-batch-size=8
```

### Multi-GPU (8× V100/A100):
```bash
uv run torchrun --nproc_per_node=8 -m scripts.looped_train \
  --run=looped_500m_v1 \
  --depth=36 \
  --save-every=1000 \
  --eval-every=250 \
  --sample-every=2000 \
  --device-batch-size=16
```

### With FP8 (H100 only — saves ~40% memory):
```bash
uv run torchrun --nproc_per_node=8 -m scripts.looped_train \
  --run=looped_500m_v1 \
  --depth=36 \
  --fp8 \
  --save-every=1000 \
  --eval-every=250 \
  --device-batch-size=32
```

## Step 7: Monitor Training

### Tmux (keep training alive after disconnect):
```bash
# Before launching training:
tmux new -s train

# Run your training command inside tmux
# To detach: Ctrl+B, then D
# To reattach: tmux attach -t train
```

### Wandb (if you want online logging):
```bash
pip install wandb
wandb login  # paste your API key
# Then use --run=looped_500m_v1 (anything except "dummy" enables wandb)
```

### Check GPU utilization:
```bash
watch -n1 nvidia-smi
```

## Step 8: Evaluate After Training

```bash
# Full evaluation (CORE metric + BPB + samples)
uv run torchrun --nproc_per_node=8 -m scripts.base_eval \
  --model-tag=looped_d36_r4 \
  --eval=core,bpb,sample
```

## Step 9: Download Checkpoint

```bash
# From your LOCAL machine:
scp -i "nanochat-key.pem" -r ubuntu@<IP>:~/nanochat/looped_checkpoints/ ./checkpoints/
```

## Step 10: Stop Instance (IMPORTANT!)

```bash
# Don't forget to stop/terminate to avoid charges!
# AWS Console → EC2 → Instances → Select → Instance State → Stop/Terminate
```

## Cost Estimates

| Instance | GPUs | Speed | Cost/hr | ~Time for 10B tokens | ~Total Cost |
|---|---|---|---|---|---|
| g5.4xlarge | 1× A10G | Slow | $1.62 | ~20 hours | ~$32 |
| p3.16xlarge | 8× V100 | Fast | $24.48 | ~4 hours | ~$98 |
| p4d.24xlarge | 8× A100 | Fastest | $32.77 | ~2 hours | ~$65 |

## Troubleshooting

- **OOM**: Reduce `--device-batch-size` (try 4, 2, or 1)
- **Tokenizer not found**: Run the training data download script first
- **NCCL errors (multi-GPU)**: Set `export NCCL_P2P_DISABLE=1` before torchrun
- **Slow training**: Check `nvidia-smi` — if GPU util < 80%, increase batch size
