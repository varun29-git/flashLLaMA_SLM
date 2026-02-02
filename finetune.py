import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
import bitsandbytes as bnb
import random

from config import *
from model import build_llama

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
FT_LR = 3e-5
FT_EPOCHS = 1 
FT_BATCH_SIZE = BATCH_SIZE
FT_SEQ_LEN = 1024
CHECKPOINT_PATH = f"{MODEL_FOLDER}/model_final.pt"
FINETUNE_MODEL_FOLDER = "checkpoints_finetune"

# --------------------------------------------------------------------------------
# Mappers
# --------------------------------------------------------------------------------
def map_smoltalk(x):
    # Keys: ['messages', ...]
    try:
        msgs = x['messages']
        text = ""
        for m in msgs:
            role = "User" if m['role'] == 'user' else "Assistant"
            text += f"{role}: {m['content']}\n\n"
        return {"text": text.strip()}
    except:
        return {"text": ""}

def map_tiny_codes(x):
    # Keys: ['prompt', 'response', metadata...] 
    # Used for code in finetuning now
    prompt = x.get('prompt')
    if not prompt:
        prompt = f"In the scenario of {x['scenario']}, write a {x['programming_language']} script for {x['target_audience']} about {x['main_topic']}."
    return {"text": f"User: {prompt}\n\nAssistant: {x['response']}"}

def map_slimorca(x):
    # Keys: ['conversations'] -> [{'from': 'system', 'value':...}, {'from': 'human'...}]
    convs = x['conversations']
    text = ""
    for c in convs:
        role = c['from']
        val = c['value']
        # Map roles
        if role == 'system':
            # Prepend system prompt to user or just label it
            text += f"System: {val}\n"
        elif role == 'human':
            text += f"User: {val}\n\n"
        elif role == 'gpt':
            text += f"Assistant: {val}\n\n"
    return {"text": text.strip()}

# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------
class MixedInstructionDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024, max_steps=50000): # 200M / (4*1024) approx 48k steps
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_steps = max_steps
        
        print("Loading datasets for mixing...")
        
        # SmolTalk (65%)
        print("   - HuggingFaceTB/smoltalk (General) (65%)")
        ds_smol = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        ds_smol = ds_smol.map(map_smoltalk)
        ds_smol = ds_smol.select_columns(["text"])
        
        # Tiny-Codes (15%)
        print("   - nampdn-ai/tiny-codes (15%)")
        ds_code = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
        ds_code = ds_code.map(map_tiny_codes)
        ds_code = ds_code.select_columns(["text"])
        
        # SlimOrca (20%) 
        print("   - Open-Orca/SlimOrca (20%)")
        ds_orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        ds_orca = ds_orca.map(map_slimorca)
        ds_orca = ds_orca.select_columns(["text"])
        
        # Interleave [0.65, 0.15, 0.20]
        probs = [0.65, 0.15, 0.20]
        self.mixed = interleave_datasets([ds_smol, ds_code, ds_orca], probabilities=probs, stopping_strategy="first_exhausted")
        self.iterator = iter(self.mixed)
        
    def __len__(self):
        # Virtual length for tqdm
        return self.max_steps
        
    def __getitem__(self, idx):
        # We ignore idx for streaming, just get next
        try:
            item = next(self.iterator)
            text = item['text']
            
            # Argilla 700 token cap check (approximate or precise)
            # We must tokenize first to check precise length
            tokens = self.tokenizer.encode(text).ids
            
            # Tokenize formatting
            eos_id = self.tokenizer.token_to_id("<EOS>")
            if eos_id is None: eos_id = 3
            tokens.append(eos_id)
            
            # Truncate to Sequence Length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # Pad
            pad_id = self.tokenizer.token_to_id("<PAD>")
            if pad_id is None: pad_id = 1
            
            padding = [pad_id] * (self.max_length - len(tokens))
            tokens = tokens + padding
            
            x = torch.tensor(tokens, dtype=torch.long)
            y = x.clone()
            if len(padding) > 0:
                y[-len(padding):] = -100
            
            return {
                "input_ids": x[:-1],
                "targets": y[1:]
            }
            
        except StopIteration:
            self.iterator = iter(self.mixed)
            return self.__getitem__(idx)

# --------------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------------
def train_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    Path(FINETUNE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Build Model
    vocab_size = tokenizer.get_vocab_size() # explicit check
    print(f"Building model (Vocab: {vocab_size})...")
    model = build_llama(
        vocab_size=vocab_size, 
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    # Load Checkpoint
    if Path(CHECKPOINT_PATH).exists():
        print(f"Loading pretrained weights from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found! Starting from scratch.")

    # Data
    # Mixed Instruction Mix: SmolTalk (65%) | SlimOrca (20%) | TinyCodes (15%)
    # Max tokens: 200M (approx 48k steps) -> 50k steps
    train_dataset = MixedInstructionDataset(tokenizer, max_length=FT_SEQ_LEN, max_steps=50000)
    train_loader = DataLoader(train_dataset, batch_size=FT_BATCH_SIZE, num_workers=1, pin_memory=True)

    # Optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=FT_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Loop
    model.train()
    print(f"Starting Fine-tuning (Mixed Strategy)...")
    
    pbar = tqdm(train_loader, desc=f"Fine-tuning", dynamic_ncols=True)
    total_loss = 0
    steps = 0
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        total_loss += loss_val
        steps += 1
        
        pbar.set_postfix({"Loss": f"{loss_val:.4f}", "Avg": f"{total_loss/steps:.4f}"})
        
        if steps % 500 == 0:
            torch.save(model.state_dict(), f"{FINETUNE_MODEL_FOLDER}/model_ft_step_{steps}.pt")
            
    print("Fine-tuning Complete.")
    torch.save(model.state_dict(), f"{FINETUNE_MODEL_FOLDER}/model_finetuned_final.pt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Mapping pickling fix: ensure mappers are importable or use 'dill', 
    # but here they are top-level so standard pickle works with num_workers>0
    train_finetune()
