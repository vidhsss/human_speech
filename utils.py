import pandas as pd
import numpy as np
import torch
import random

def extract_vocab(csv_path):
    df = pd.read_csv(csv_path)
    vocab = set()
    for t in df['transcription']:
        vocab.update(list(t))
    vocab = sorted(list(vocab))
    vocab_dict = {c: i+1 for i, c in enumerate(vocab)}  # 0 is reserved for blank in CTC
    vocab_dict['<blank>'] = 0
    inv_vocab_dict = {i: c for c, i in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict

def text_to_indices(text, vocab_dict):
    return [vocab_dict[c] for c in text if c in vocab_dict]

def indices_to_text(indices, inv_vocab_dict):
    return ''.join([inv_vocab_dict[i] for i in indices if i in inv_vocab_dict and i != 0])

def add_noise(audio, noise_level=0.05):
    noise = torch.randn_like(audio) * noise_level
    return audio + noise 