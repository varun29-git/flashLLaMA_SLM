
import torch
from model import build_llama
from config import *
from datasets import load_dataset
from dataset import StreamingLanguageModelDataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

def verify():
    print("Verifying Model Configuration...")
    # Expect VOCAB_SIZE = 4096
    vocab_size = VOCAB_SIZE 
    print(f"Vocab Size: {vocab_size}")
    
    model = build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    # Expected: ~2.5M
    # Embedding: 32000 * 64 = 2,048,000
    # Layers: ~0.5M
    # Total roughly ~2.5M

    
    print("\nVerifying Tokenizer & Dataset Loading...")
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
        print("Loaded tokenizer.json")
        
        ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
        dl = DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), batch_size=1)
        batch = next(iter(dl))
        print("Successfully loaded one batch from Cosmopedia using custom tokenizer.")
        print(f"Input shape: {batch['input_ids'].shape}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    verify()

if __name__ == "__main__":
    verify()
