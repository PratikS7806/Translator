import os
from datasets import load_dataset

print("Downloading OPUS-100 dataset...")
dataset = load_dataset(
    "Helsinki-NLP/opus-100", 
    "de-en", 
    split="train",
    cache_dir="D:/AIMDM/Transformer/data/cache"
)

# Take the first 100,000 sentence pairs
SUBSET_SIZE = 100000
print(f"Sampling {SUBSET_SIZE} sentences for fast, diverse conversational training...")

with open("data/multi30k/train.de", "a", encoding="utf-8") as f_de, \
     open("data/multi30k/train.en", "a", encoding="utf-8") as f_en:
    
    for i in range(min(SUBSET_SIZE, len(dataset))):
        translation = dataset[i]["translation"]
        de_text = translation["de"].replace("\n", " ").strip()
        en_text = translation["en"].replace("\n", " ").strip()
        
        # Only write non-empty lines
        if de_text and en_text:
            f_de.write("\n" + de_text)
            f_en.write("\n" + en_text)

print(f"Successfully added {SUBSET_SIZE} conversational pairs to Multi30k!")