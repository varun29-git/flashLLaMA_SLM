import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import warnings
import time
import math

import tiktoken

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset


# ============================================================
# Model builder
# ============================================================
def get_model(vocab_size):
    return build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )


# ============================================================
# Simple text sampling (sanity check)
# ============================================================
@torch.no_grad()
def sample_text(model, tokenizer, device, prompt, max_new_tokens=80):
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(x)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# ============================================================
# One epoch of training (NO tqdm, stable)
# ============================================================
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epoch,
    global_step,
    total_steps,
    token_counter,
    run_start_time,
):
    model.train()
    epoch_start_time = time.time()

    for step, batch in enumerate(dataloader):
        if step >= STEPS_PER_EPOCH:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- accounting ----
        batch_tokens = input_ids.numel()
        token_counter[0] += batch_tokens
        global_step += 1

        # ---- logging ----
        if global_step % LOG_INTERVAL == 0 or global_step == 1:
            elapsed = time.time() - run_start_time
            tok_per_sec = token_counter[0] / max(elapsed, 1e-6)

            progress = global_step / total_steps
            percent = progress * 100

            steps_left = total_steps - global_step
            eta_seconds = steps_left * (elapsed / max(global_step, 1))
            eta_hours = eta_seconds / 3600

            lr = scheduler.get_last_lr()[0]

            print(
                f"[epoch {epoch+1}/{EPOCHS}] "
                f"{percent:6.2f}% | "
                f"step {global_step}/{total_steps} | "
                f"loss={loss.item():.4f} | "
                f"lr={lr:.2e} | "
                f"tok/s={tok_per_sec/1000:.2f}k | "
                f"ETA={eta_hours:.2f}h"
            )

    return global_step


# ============================================================
# Main training entry
# ============================================================
def train(iterable_ds):
    # --------------------
    # Device
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    # --------------------
    # Dataset
    # --------------------
    print("Datasets:")
    print(" - TinyStories (streaming)")
    print(" - OpenWebText (streaming)")
    print("Mixing ratio: 60% / 40%\n")

    train_dataset = StreamingLanguageModelDataset(
        iterable_ds,
        seq_len=SEQ_LEN,
        tokenizer_name="cl100k_base"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,   # REQUIRED for IterableDataset
        pin_memory=True
    )

    # --------------------
    # Model
    # --------------------
    vocab_size = 100277  # cl100k_base
    model = get_model(vocab_size).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # --------------------
    # Tokenizer + sample
    # --------------------
    tokenizer = tiktoken.get_encoding("cl100k_base")

    print("🔍 Sample generation BEFORE training:\n")
    print(
        sample_text(
            model,
            tokenizer,
            device,
            prompt="Once upon a time",
            max_new_tokens=60
        )
    )
    print("\n" + "=" * 80 + "\n")

    # --------------------
    # Optimizer / loss / scheduler
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps = EPOCHS * STEPS_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    # --------------------
    # Training
    # --------------------
    print(f"Starting Training | {EPOCHS} epochs | {total_steps:,} steps")
    print("=" * 80)

    global_step = 0
    token_counter = [0]
    run_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)

        global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            epoch,
            global_step,
            total_steps,
            token_counter,
            run_start_time,
        )

        ckpt = Path(MODEL_FOLDER) / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "tokens_seen": token_counter[0],
            },
            ckpt
        )
        print(f"✓ Checkpoint saved: {ckpt}")

    # --------------------
    # Final summary
    # --------------------
    total_time_hours = (time.time() - run_start_time) / 3600
    total_tokens = token_counter[0]
    chinchilla_ratio = total_tokens / n_params

    print("\n" + "=" * 80)
    print("📊 Training Summary")
    print(f"Total parameters : {n_params:,}")
    print(f"Total tokens     : {total_tokens:,}")
    print(f"Chinchilla ratio : {chinchilla_ratio:.2f} tokens/param")
    print(f"Total time       : {total_time_hours:.2f} hours")
    print("=" * 80)

    return model


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    from datasets import load_dataset, interleave_datasets

    ds_tiny = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds_openweb = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    mixed_iterable = interleave_datasets(
        [ds_tiny, ds_openweb],
        probabilities=[0.6, 0.4],
        seed=42
    )

    train(mixed_iterable)
