
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import BpeTrainer

def train_tokenizer():
    print("Loading dataset samples for tokenizer training...")
    # Cosmopedia
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
    # FineWeb-Edu
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    # Tiny-Codes
    ds_code = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
    # DCLM
    ds_dclm = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)

    # Create an iterator that yields text from all four 
    def batch_iterator(batch_size=1000):
        buffer = []
        
        # Take 10000 examples from Cosmo
        print("Sampling Cosmopedia (10k)...")
        for i, item in enumerate(ds_cosmo):
            if i >= 10000: break
            buffer.append(item['text'])

        # Take 6000 examples from FW-Edu
        print("Sampling FineWeb-Edu (6k)...")
        for i, item in enumerate(ds_fw):
            if i >= 6000: break
            buffer.append(item['text'])
            
        # Take 2000 examples from Code
        print("Sampling Tiny-Codes (2k)...")
        for i, item in enumerate(ds_code):
            if i >= 2000: break
            # Map prompt/response
            prompt = item.get('prompt')
            if not prompt:
                prompt = f"In the scenario of {item.get('scenario','')}, write a {item.get('programming_language','')} script for {item.get('target_audience','')} about {item.get('main_topic','')}."
            text = f"{prompt}\n{item.get('response', '')}"
            buffer.append(text)

        # Take 2000 examples from DCLM
        print("Sampling DCLM (2k)...")
        for i, item in enumerate(ds_dclm):
            if i >= 2000: break
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
        vocab_size=16384,
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

