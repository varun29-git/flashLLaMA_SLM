import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import warnings
import time
import math
from collections import deque

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset


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
    run_start_time,
):
    model.train()

    # Rolling stats
    loss_window = deque(maxlen=50)
    epoch_start_time = time.time()
    tokens_seen = 0

    pbar = tqdm(
    total=STEPS_PER_EPOCH,
    desc=f"Epoch {epoch+1}/{EPOCHS}",
    dynamic_ncols=True,
    leave=True
)

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

        # ---- stats ----
        loss_window.append(loss.item())
        tokens_seen += input_ids.numel()
        global_step += 1

        elapsed = time.time() - epoch_start_time
        run_elapsed = time.time() - run_start_time

        tok_per_sec = tokens_seen / max(elapsed, 1e-6)
        avg_loss = sum(loss_window) / len(loss_window)
        lr = scheduler.get_last_lr()[0]

        steps_left = total_steps - global_step
        eta_seconds = steps_left * (run_elapsed / max(global_step, 1))
        eta_hours = eta_seconds / 3600

        pbar.set_postfix({
            "loss": f"{avg_loss:6.3f}",
            "lr": f"{lr:.2e}",
            "tok/s": f"{tok_per_sec/1000:.1f}k",
            "ETA(h)": f"{eta_hours:5.2f}"
        })
        pbar.update(1)
    
    pbar.close()

    return global_step


def train(iterable_ds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    train_dataset = StreamingLanguageModelDataset(
        iterable_ds,
        seq_len=SEQ_LEN,
        tokenizer_name="cl100k_base"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,   # correct for IterableDataset
        pin_memory=True
    )

    vocab_size = 100277  # cl100k_base
    model = get_model(vocab_size).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

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

    global_step = 0
    run_start_time = time.time()

    print("\n" + "=" * 70)
    print(f"Starting Training | {EPOCHS} epochs | {total_steps:,} steps")
    print("=" * 70)

    for epoch in range(EPOCHS):
        print("\n" + "-" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS}")
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
            run_start_time
        )

        ckpt = Path(MODEL_FOLDER) / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            ckpt
        )
        print(f"✓ Checkpoint saved: {ckpt}")

    total_hours = (time.time() - run_start_time) / 3600
    print("\n" + "=" * 70)
    print(f"Training complete in {total_hours:.2f} hours")
    print("=" * 70)

    return model


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
