import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from collections import deque
from datasets import load_dataset

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset
import random
import math

ESTIMATED_TOTAL_TOKENS = 1_675_000_000
LR_PHASE_1 = 2e-4

# Constants for Schedule
PHASE1_DURATION = 525_000_000
PHASE2_DURATION = 1_150_000_000
TOTAL_TRAINING_TOKENS = PHASE1_DURATION + PHASE2_DURATION

def get_lr(tokens_seen):
    # Phase 1: 0 - 625M
    if tokens_seen < PHASE1_DURATION:
        # Constant 2e-4 for first 180M
        if tokens_seen < 180_000_000:
            return 2e-4
        
        # Cosine Decay 2e-4 -> 2e-5 (180M to 625M)
        progress = (tokens_seen - 180_000_000) / (PHASE1_DURATION - 180_000_000)
        progress = max(0.0, min(1.0, progress))
        # decay from 2e-4 to 2e-5
        
        target_min = 2e-5
        target_max = 2e-4
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return target_min + (target_max - target_min) * cosine_decay

    # Phase 2: 625M - End
    else:
        phase2_tokens = tokens_seen - PHASE1_DURATION
        
        # Linear Warmup 2e-5 -> 1e-4 (over 50M tokens)
        if phase2_tokens < 50_000_000:
            progress = phase2_tokens / 50_000_000
            start_lr = 2e-5
            end_lr = 1e-4
            return start_lr + (end_lr - start_lr) * progress
            
        # Cosine Decay 1e-4 -> 1e-5 (remaining 1.15B)
        else:
            decay_tokens = phase2_tokens - 50_000_000
            total_decay_duration = PHASE2_DURATION - 50_000_000
            progress = decay_tokens / total_decay_duration
            progress = max(0.0, min(1.0, progress))
            
            target_min = 1e-5
            target_max = 1e-4
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return target_min + (target_max - target_min) * cosine_decay  

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

def validate(model, dataset_name, device, steps=50):
    """Runs a quick validation loop on the validation or test split."""
    print(f"\n--- Running Validation for {dataset_name} ---")
    model.eval()
    
    ds = None
    try:
        # Try validation split first
        ds = load_dataset(dataset_name, split="validation", streaming=True)
    except Exception:
        try:
            # Fallback to test split
            ds = load_dataset(dataset_name, split="test", streaming=True)
        except Exception:
            print(f"No validation/test split found for {dataset_name}. Skipping validation.")
            model.train()
            return None

    val_dataset = StreamingLanguageModelDataset(
        ds,
        seq_len=SEQ_LEN,
        tokenizer_name="cl100k_base"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    val_loss_accum = 0.0
    steps_done = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            for i, batch in enumerate(val_loader):
                if i >= steps:
                    break
                
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                
                logits = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                val_loss_accum += loss.item()
                steps_done += 1
    
    avg_val_loss = val_loss_accum / max(steps_done, 1)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    model.train()
    return avg_val_loss


def train_mixed_phase_1(model, optimizer, scaler, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Configuration
    total_duration = PHASE1_DURATION
    mix_decay_duration = total_duration # Linear decay over whole phase
    
    phase_name = "Phase 1: Mixed (TS 60%->10% | GB 40%->90%)"
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name}")
    print(f"Duration: {total_duration:,} tokens")
    print("=" * 80)

    # Initialize Datasets
    ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds_gb = load_dataset("incredible45/Gutenberg-BookCorpus-Cleaned-Data-English", split="train", streaming=True).rename_column("context", "text")

    dl_ts = DataLoader(StreamingLanguageModelDataset(ds_ts, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)
    dl_gb = DataLoader(StreamingLanguageModelDataset(ds_gb, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)

    iter_ts = iter(dl_ts)
    iter_gb = iter(dl_gb)

    total_phase_tokens = 0
    
    # Progress Bar
    pbar = tqdm(total=total_duration // (BATCH_SIZE * SEQ_LEN), dynamic_ncols=True)
    
    model.train()
    loss_window = deque(maxlen=50)
    optimizer.zero_grad(set_to_none=True)
    
    step = 0
    current_lr = 0.0
    
    while total_phase_tokens < total_duration:
        step += 1
        
        # Calculate Learning Rate
        current_lr = get_lr(global_tracker['tokens_seen'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Calculate Mix Ratio (60% -> 10%)
        progress = total_phase_tokens / mix_decay_duration
        progress = max(0.0, min(1.0, progress))
        p_ts = 0.6 - (0.5 * progress) # Starts at 0.6, ends at 0.1
            
        # Select Batch
        use_ts = random.random() < p_ts
        
        try:
            if use_ts:
                batch = next(iter_ts)
            else:
                batch = next(iter_gb)
        except StopIteration:
            if use_ts:
                iter_ts = iter(dl_ts)
                batch = next(iter_ts)
            else:
                iter_gb = iter(dl_gb)
                batch = next(iter_gb)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        
        batch_tokens = input_ids.numel()
        total_phase_tokens += batch_tokens
        pbar.update(1)
        
        if global_tracker:
            global_tracker['tokens_seen'] += batch_tokens

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)
        
        # ETA
        eta_str = "??"
        if global_tracker:
             elapsed = time.time() - global_tracker['start_time']
             rate = global_tracker['tokens_seen'] / max(elapsed, 1e-6)
             remaining = ESTIMATED_TOTAL_TOKENS - global_tracker['tokens_seen']
             eta_seconds = remaining / max(rate, 1e-6)
             eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"

        pbar.set_postfix({
            "TS": f"{p_ts:.0%}",
            "LR": f"{current_lr:.1e}",
            "L": f"{avg_loss:.2f}",
            "ETA": eta_str
        })

    pbar.close()
    print(f"Phase 1 Complete. Tokens: {total_phase_tokens:,}")
    validate(model, "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English", device)


def train_phase_2(model, optimizer, scaler, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Configuration
    total_duration = PHASE2_DURATION
    
    phase_name = "Phase 2: FineWeb + TS (Fixed 10%)"
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name}")
    print(f"Duration: {total_duration:,} tokens")
    print("=" * 80)

    # Initialize Datasets
    ds_main = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    dl_main = DataLoader(StreamingLanguageModelDataset(ds_main, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)
    dl_ts = DataLoader(StreamingLanguageModelDataset(ds_ts, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)

    iter_main = iter(dl_main)
    iter_ts = iter(dl_ts)

    total_phase_tokens = 0
    pbar = tqdm(total=total_duration // (BATCH_SIZE * SEQ_LEN), dynamic_ncols=True)
    
    model.train()
    loss_window = deque(maxlen=50)
    optimizer.zero_grad(set_to_none=True)
    
    step = 0
    current_lr = 0.0
    
    while total_phase_tokens < total_duration:
        step += 1
        
        current_lr = get_lr(global_tracker['tokens_seen'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Fixed Ratio 10%
        p_ts = 0.10
        use_ts = random.random() < p_ts
        
        try:
            if use_ts:
                batch = next(iter_ts)
            else:
                batch = next(iter_main)
        except StopIteration:
            if use_ts:
                iter_ts = iter(dl_ts)
                batch = next(iter_ts)
            else:
                iter_main = iter(dl_main)
                batch = next(iter_main)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        
        batch_tokens = input_ids.numel()
        total_phase_tokens += batch_tokens
        pbar.update(1)
        
        if global_tracker:
            global_tracker['tokens_seen'] += batch_tokens

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)
        
        eta_str = "??"
        if global_tracker:
             elapsed = time.time() - global_tracker['start_time']
             rate = global_tracker['tokens_seen'] / max(elapsed, 1e-6)
             remaining = ESTIMATED_TOTAL_TOKENS - global_tracker['tokens_seen']
             eta_seconds = remaining / max(rate, 1e-6)
             eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"

        pbar.set_postfix({
            "LR": f"{current_lr:.1e}",
            "L": f"{avg_loss:.2f}",
            "ETA": eta_str
        })

    pbar.close()
    print(f"Phase 2 Complete. Tokens: {total_phase_tokens:,}")
    validate(model, "HuggingFaceFW/fineweb-edu", device)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    vocab_size = 100277
    model = get_model(vocab_size).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE_1,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    # Global Progress Tracker
    global_tracker = {
        'start_time': time.time(),
        'tokens_seen': 0
    }

    # MIXED PHASE 1 (TS + Gutenberg)
    train_mixed_phase_1(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        vocab_size=vocab_size,
        global_tracker=global_tracker
    )
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase1.pt")

    # MIXED PHASE 2 (FineWeb + TS)
    train_phase_2(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        vocab_size=vocab_size,
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/model_final.pt")
    
    total_time = time.time() - global_tracker['start_time']
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Total Time: {total_time/3600:.2f} hours")
    print(f"Total Tokens Processed: {global_tracker['tokens_seen']:,}")
    print("=" * 80)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
