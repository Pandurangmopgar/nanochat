"""
Train a Looped Transformer model. From root directory of the project, run as:

python -m scripts.looped_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.looped_train

If you are only on CPU/Macbook, train a tiny model:
python -m scripts.looped_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

This is a fork of base_train.py adapted for the Looped Transformer (LoopedGPT).
Key differences:
- Uses LoopedGPT instead of GPT
- Adds --mean-recurrence, --max-recurrence, --backprop-depth, --use-lora, --use-mod CLI args
- Logs recurrence info (n_steps, MoD capacity) per training step
- Uses LoopedGPTConfig instead of GPTConfig
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
import torch

from nanochat.looped_gpt import LoopedGPT, LoopedGPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain Looped Transformer model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")
# Model architecture
parser.add_argument("--depth", type=int, default=36, help="equivalent depth (used for model_dim = depth * aspect_ratio)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")

# Looped Transformer specific
parser.add_argument("--n-prelude", type=int, default=2, help="number of prelude layers")
parser.add_argument("--n-recurrent", type=int, default=3, help="number of layers in recurrent block")
parser.add_argument("--n-coda", type=int, default=1, help="number of coda layers")
parser.add_argument("--mean-recurrence", type=int, default=4, help="mean recurrence during training (r̄)")
parser.add_argument("--max-recurrence", type=int, default=16, help="maximum recurrence (safety cap)")
parser.add_argument("--backprop-depth", type=int, default=4, help="truncated backprop through last k iterations")
parser.add_argument("--state-init-std", type=float, default=0.02, help="std for random state initialization")
parser.add_argument("--use-lora", action="store_true", default=True, help="enable LoRA depth adapters")
parser.add_argument("--no-lora", action="store_false", dest="use_lora", help="disable LoRA depth adapters")
parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank per depth adapter")
parser.add_argument("--use-mod", action="store_true", default=True, help="enable Mixture-of-Depths routing")
parser.add_argument("--no-mod", action="store_false", dest="use_mod", help="disable Mixture-of-Depths routing")
parser.add_argument("--embedding-scale", type=float, default=1.0, help="embedding scale (γ)")

# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="data:param ratio (Chinchilla=20)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="LR for embedding (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for unembedding (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for Muon")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="LR for scalars")
parser.add_argument("--adam-beta1", type=float, default=0.8)
parser.add_argument("--adam-beta2", type=float, default=0.95)
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume from step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="eval val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="tokens for val evaluation")
parser.add_argument("--core-metric-every", type=int, default=2000, help="eval CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500)
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint dir")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process or not HAS_WANDB
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-looped", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    print0("!" * 80)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Looped Model

def build_looped_model_meta():
    """Build a LoopedGPT model on meta device."""
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim

    config = LoopedGPTConfig(
        vocab_size=vocab_size,
        sequence_len=args.max_seq_len,
        n_embd=model_dim,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_prelude=args.n_prelude,
        n_recurrent=args.n_recurrent,
        n_coda=args.n_coda,
        mean_recurrence=args.mean_recurrence,
        max_recurrence=args.max_recurrence,
        backprop_depth=args.backprop_depth,
        state_init_std=args.state_init_std,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        use_mod=args.use_mod,
        embedding_scale=args.embedding_scale,
    )
    with torch.device("meta"):
        model_meta = LoopedGPT(config)
    return model_meta

model = build_looped_model_meta()
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"\n{'='*60}")
print0(f"LOOPED TRANSFORMER Configuration:")
print0(f"{'='*60}")
print0(f"P-R-C structure: ({model_config.n_prelude}, {model_config.n_recurrent}, {model_config.n_coda})")
print0(f"Mean recurrence: {model_config.mean_recurrence}")
print0(f"Max recurrence: {model_config.max_recurrence}")
print0(f"Backprop depth: {model_config.backprop_depth}")
print0(f"LoRA: {'enabled (rank=' + str(model_config.lora_rank) + ')' if model_config.use_lora else 'disabled'}")
print0(f"MoD: {'enabled' if model_config.use_mod else 'disabled'}")
print0(f"Embedding dim: {model_config.n_embd}")
print0(f"Heads: {model_config.n_head}")
print0(f"{'='*60}\n")
print0(f"Full config:\n{json.dumps(model_config_kwargs, indent=2)}")

model.to_empty(device=device)
model.init_weights()

# Checkpoint handling
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"looped_d{args.depth}_r{args.mean_recurrence}"
checkpoint_dir = os.path.join(base_dir, "looped_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# FP8 training (same as base_train.py)
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 training enabled ({args.fp8_recipe}) - converted {num_fp8_layers} layers, skipped {num_skipped}")

@contextmanager
def disable_fp8(model):
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))
    if not fp8_locations:
        yield
        return
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(fp8_module.in_features, fp8_module.out_features,
                          bias=fp8_module.bias is not None, device=fp8_module.weight.device, dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model
orig_model = model
model = torch.compile(model, dynamic=False)

# Parameter counts
param_counts = model.num_params()
print0(f"\nParameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")
num_params = param_counts['total']

# Effective depth for FLOPs estimation
effective_layers = model_config.n_prelude + model_config.n_recurrent * model_config.mean_recurrence + model_config.n_coda
approx_flops_per_token = 6 * num_params  # rough estimate
print0(f"Effective depth at r̄={model_config.mean_recurrence}: {effective_layers} layers")
print0(f"Approx FLOPs per token: {approx_flops_per_token:e}")
num_flops_per_token = approx_flops_per_token

# Scaling — use recurrent_block + lm_head for scaling params (like base_train)
num_scaling_params = param_counts['recurrent_block'] + param_counts['lm_head']
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Batch size and LR scaling (simplified from base_train — using d12 reference)
from nanochat.gpt import GPT, GPTConfig
d12_config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
with torch.device("meta"):
    d12_ref = GPT(d12_config)
d12_scaling = sum(p.numel() for p in d12_ref.transformer.h.parameters()) + sum(p.numel() for p in d12_ref.lm_head.parameters())
D_REF = args.target_param_data_ratio * d12_scaling
B_REF = 2**19

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed batch size: {total_batch_size:,} tokens")

batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f}")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Initialize Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Training iterations
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Iterations from data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(f"Tokens:Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")

# Schedulers
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Loop state
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => grad accum steps: {grad_accum_steps}")

# Training loop
while True:
    last_step = step == num_iterations

    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "val/bpb": val_bpb})
        model.train()

    # CORE metric
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "core_metric": results["core_metric"], "centered_results": results["centered_results"]})
        model.train()

    # Sampling
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "If 5*x + 3 = 13, then x is",
            "The opposite of hot is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(), optimizer.state_dict(),
            {
                "step": step, "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Update optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })

    # GC management
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# Final stats
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Min validation bpb: {min_val_bpb:.6f}")

# Report
from nanochat.report import get_report
get_report().log(section="Looped Transformer training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "P-R-C structure": f"({model_config.n_prelude}, {model_config.n_recurrent}, {model_config.n_coda})",
        "Mean recurrence": model_config.mean_recurrence,
        "LoRA enabled": model_config.use_lora,
        "MoD enabled": model_config.use_mod,
        "Effective depth": effective_layers,
        "Number of iterations": num_iterations,
        "Total training tokens": total_tokens,
        "Tokens:Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
    },
    {
        "Min val bpb": min_val_bpb if val_bpb is not None else None,
        "Final val bpb": val_bpb,
        "CORE metric": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

wandb_run.finish()
compute_cleanup()
