import os
import urllib.request
import gzip
import shutil
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def setup_multi30k_and_tokenizer():
    os.makedirs("data/multi30k", exist_ok=True)
    urls = {
        "train.de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz",
        "train.en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz"
    }

    # Download and extract
    for name, url in urls.items():
        file_path = f"data/multi30k/{name}.gz"
        out_path = f"data/multi30k/{name}"
        if not os.path.exists(out_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, file_path)
            with gzip.open(file_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Combine data for shared vocabulary
    combined_file = "data/multi30k/train.combined"
    if not os.path.exists(combined_file):
        with open(combined_file, "w", encoding="utf-8") as outfile:
            for lang in ["train.de", "train.en"]:
                with open(f"data/multi30k/{lang}", "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())

    # Train SentencePiece if it doesn't exist
    if not os.path.exists("m30k_bpe.model"):
        print("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(
            input=combined_file,
            model_prefix='m30k_bpe',
            vocab_size=32000,
            model_type='bpe',
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>'
        )
    return spm.SentencePieceProcessor(model_file='m30k_bpe.model')

def get_dataloader(batch_size=32, max_seq_len=64):
    sp = setup_multi30k_and_tokenizer()
    PAD_IDX = sp.pad_id()
    BOS_IDX = sp.bos_id()
    EOS_IDX = sp.eos_id()

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_text, tgt_text in batch:
            # Truncate tokens to ensure sequence length stays bounded
            src_ids = sp.encode_as_ids(src_text)[: max_seq_len - 2]
            tgt_ids = sp.encode_as_ids(tgt_text)[: max_seq_len - 2]

            src_encoded = [BOS_IDX] + src_ids + [EOS_IDX]
            tgt_encoded = [BOS_IDX] + tgt_ids + [EOS_IDX]

            src_batch.append(torch.tensor(src_encoded))
            tgt_batch.append(torch.tensor(tgt_encoded))

        src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_padded, tgt_padded

    with open("data/multi30k/train.de", "r", encoding="utf-8") as f:
        de_lines = [line.strip() for line in f]
    with open("data/multi30k/train.en", "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]

    train_dataset = list(zip(de_lines, en_lines))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader, sp