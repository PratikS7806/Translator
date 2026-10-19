# Basic Transformer From Scratch

A lean, from-scratch implementation of the original encoder-decoder Transformer architecture using plain PyTorch. Built to translate German to English using the Multi30k dataset.

## Architecture Highlights
- **Core Components:** Hand-written Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding, and Feed-Forward Networks.
- **Variant Used:** Pre-LayerNorm architecture (`x + Sublayer(LN(x))`) for superior training stability.
- **Tokenizer:** 8,000-token shared Byte-Pair Encoding (BPE) dictionary trained via `sentencepiece`.
- **Model Size:** 10 Million parameters (`d_model`=256, `heads`=4, `layers`=3).

## Quickstart

**1. Install Dependencies**
```bash
pip install torch sentencepiece sacrebleu