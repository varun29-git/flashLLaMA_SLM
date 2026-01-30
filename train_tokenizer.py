
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import BpeTrainer

def train_tokenizer():
    print("Loading dataset samples for tokenizer training...")
    # Load separate iterators to avoid mixing logic issues
    ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)

    # Create an iterator that yields text from both
    def batch_iterator(batch_size=1000):
        buffer = []
        # Take 5000 examples from TS
        print("Sampling TinyStories...")
        for i, item in enumerate(ds_ts):
            if i >= 5000: break
            buffer.append(item['text'])
        
        # Take 5000 examples from Cosmo
        print("Sampling Cosmopedia...")
        for i, item in enumerate(ds_cosmo):
            if i >= 5000: break
            buffer.append(item['text'])
            
        print(f"Collected {len(buffer)} samples. Training tokenizer...")
        
        for i in range(0, len(buffer), batch_size):
            yield buffer[i : i + batch_size]

    # Initialize Tokenizer (BPE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=32000,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")
    
    # Verification
    print("\nVerification:")
    output = tokenizer.encode("Hello, this is a test.")
    print(f"Encoded 'Hello, this is a test.': {output.ids}")
    print(f"Decoded: {tokenizer.decode(output.ids)}")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_tokenizer()
