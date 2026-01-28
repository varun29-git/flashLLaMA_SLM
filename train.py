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

ESTIMATED_TOTAL_TOKENS = 2_950_000_000

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

def train_phase(model, optimizer, scaler, dataset_name, phase_name, num_epochs, target_lr, vocab_size, max_tokens=None, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Update Learning Rate for this phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
        
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {num_epochs} (Logical Passes)")
    print(f"Learning Rate: {target_lr}")
    if max_tokens:
        print(f"Token Cap: {max_tokens:,}")
    else:
        print("Token Cap: None (Full Dataset Phase)")
    print("=" * 80)

    total_phase_tokens = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} of {phase_name} ---")
        
        # Load Dataset Stream (Restarted each epoch)
        try:
            if dataset_name == "bookcorpus" or dataset_name == "rojagtap/bookcorpus":
                ds = load_dataset(dataset_name, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
            
            # Show Sample
            print(f"\n[Sample from {dataset_name}]")
            try:
                sample_item = next(iter(ds))
                print(f"{sample_item['text'][:300]}...\n")
            except Exception as e:
                print(f"Could not fetch sample: {e}")

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return

        train_dataset = StreamingLanguageModelDataset(
            ds,
            seq_len=SEQ_LEN,
            tokenizer_name="cl100k_base"
        )
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=True
        )

        model.train()
        loss_window = deque(maxlen=50)
        
        # bar_format string to match user request: "Processing Epoch00: 100%|...| [time, rate, postfix]"
        # Hides the "n/total" part.
        pbar = tqdm(
            dataloader, 
            desc=f"Processing Epoch{epoch:02d}", 
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
            # --- TOKEN ACCOUNTING ---
            batch_tokens = input_ids.numel()
            
            # Phase Cap Check
            if max_tokens is not None and (total_phase_tokens + batch_tokens > max_tokens):
                print(f"\n[STOP] Token Cap Reached for {phase_name}: {total_phase_tokens + batch_tokens:,} > {max_tokens:,}")
                return  # End Phase Immediately
            
            total_phase_tokens += batch_tokens
            
            # Global Tracker Update
            if global_tracker:
                global_tracker['tokens_seen'] += batch_tokens

            # --- OPTIMIZATION STEP ---
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
            pbar.set_postfix_str(f"loss:={avg_loss:.3f}, G-ETA={eta_str}")
            
        print(f"Epoch {epoch+1} Complete. Tokens so far: {total_phase_tokens:,}")


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
        max_tokens=100, # CAP AT 100 TOKENS (TESTING)
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase1.pt")

    # PHASE 2

    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="rojagtap/bookcorpus", 
        phase_name="Phase 2 (BookCorpus)",
        num_epochs=2,
        target_lr=LR_PHASE_2,
        vocab_size=vocab_size,
        max_tokens=None, # Use Full Dataset
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase2.pt")

    # PHASE 3
    
    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="Skylion007/openwebtext",
        phase_name="Phase 3 (OpenWebText)",
        num_epochs=1,
        target_lr=LR_PHASE_3,
        vocab_size=vocab_size,
        max_tokens=500_000_000, # CAP AT 500 MILLION
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
