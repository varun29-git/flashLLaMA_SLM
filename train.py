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

ESTIMATED_TOTAL_TOKENS = 2_970_000_000

LR_PHASE_1 = 3e-4  
LR_PHASE_2 = 1e-4  
LR_PHASE_3 = 5e-5  

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


def train_phase(model, optimizer, scaler, dataset_name, phase_name, num_epochs, target_lr, vocab_size, max_tokens=None, global_tracker=None, soft_cap=False):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Update Learning Rate 
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
        
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {num_epochs} (Logical Passes)")
    print(f"Learning Rate: {target_lr}")
    if max_tokens:
        cap_type = "Soft (Finish Book)" if soft_cap else "Hard (Immediate Stop)"
        print(f"Token Cap: {max_tokens:,} [{cap_type}]")
    else:
        print("Token Cap: None (Full Dataset Phase)")
    print("=" * 80)

    total_phase_tokens = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} of {phase_name} ---")
        
        # Load Dataset Stream (Restarted each epoch)
        try:
            if dataset_name == "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English":
                ds = load_dataset(dataset_name, split="train", streaming=True)
                # Map 'context' to 'text' 
                ds = ds.rename_column("context", "text")
            elif dataset_name == "HuggingFaceFW/fineweb-edu":
                # Use the sample-10BT 
                ds = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
            

            
            # Sample display removed as requested

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return

        # Pass max_tokens to dataset ONLY if soft_cap is True
        ds_max_tokens = max_tokens if soft_cap else None
        
        train_dataset = StreamingLanguageModelDataset(
            ds,
            seq_len=SEQ_LEN,
            tokenizer_name="cl100k_base",
            max_tokens=ds_max_tokens
        )
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=True
        )

        model.train()
        loss_window = deque(maxlen=50)
        
        pbar = tqdm(
            dataloader, 
            desc=f"Processing Epoch{epoch:02d}", 
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
            batch_tokens = input_ids.numel()
            
            # Hard Cap Check: Only if soft_cap is False
            if not soft_cap and max_tokens is not None and (total_phase_tokens + batch_tokens > max_tokens):
                print(f"\n[STOP] Token Cap Reached for {phase_name}: {total_phase_tokens + batch_tokens:,} > {max_tokens:,}")
                return  # End Phase Immediately
            
            total_phase_tokens += batch_tokens
            
            # Global Tracker Update
            if global_tracker:
                global_tracker['tokens_seen'] += batch_tokens

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- STATS ---
            loss_window.append(loss.item())
            avg_loss = sum(loss_window) / len(loss_window)
            
            # Global ETA Calculation
            eta_str = "??"
            if global_tracker:
                elapsed = time.time() - global_tracker['start_time']
                rate = global_tracker['tokens_seen'] / max(elapsed, 1e-6)
                remaining = ESTIMATED_TOTAL_TOKENS - global_tracker['tokens_seen']
                eta_seconds = remaining / max(rate, 1e-6)
                eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"

            pbar.set_description(f"Processing Epoch{epoch:02d}")
            
        print(f"Epoch {epoch+1} Complete. Tokens so far: {total_phase_tokens:,}")
        
        # Validation Step
        validate(model, dataset_name, device)


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

    # PHASE 1

    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="roneneldan/TinyStories",
        phase_name="Phase 1 (TinyStories)",
        num_epochs=1,
        target_lr=LR_PHASE_1,
        vocab_size=vocab_size,
        max_tokens=None, # Use Full Dataset (Reverted Testing Cap)
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase1.pt")

    # PHASE 2

    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="incredible45/Gutenberg-BookCorpus-Cleaned-Data-English", 
        phase_name="Phase 2 (Gutenberg-BookCorpus-Cleaned-Data-English)",
        num_epochs=1, # Soft Cap controls duration within epoch
        target_lr=LR_PHASE_2,
        vocab_size=vocab_size,
        max_tokens=500_000_000, # SOFT CAP AT 500 MILLION
        global_tracker=global_tracker,
        soft_cap=True
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase2.pt")

    # PHASE 3
    

    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="HuggingFaceFW/fineweb-edu",
        phase_name="Phase 3 (FineWeb-Edu 1B)",
        num_epochs=2,
        target_lr=LR_PHASE_3,
        vocab_size=vocab_size,
        max_tokens=1_000_000_000, # 1B per epoch 
        global_tracker=global_tracker,
        soft_cap=True
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
