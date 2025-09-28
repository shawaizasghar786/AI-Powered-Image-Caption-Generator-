# utils/build_vocab.py
import nltk
import json
from collections import Counter

def build_vocab(caption_file, threshold=5):
    with open(caption_file, 'r') as f:
        lines = f.readlines()

    counter = Counter()
    captions = []
    for line in lines:
        if '\t' in line:
            _, caption = line.strip().split('\t')
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            captions.append(tokens)

    words = [w for w, c in counter.items() if c >= threshold]
    vocab = {w: i+1 for i, w in enumerate(words)}
    vocab['<PAD>'] = 0
    vocab['<START>'] = len(vocab)
    vocab['<END>'] = len(vocab) + 1
    vocab['<UNK>'] = len(vocab) + 2

    with open("utils/vocab.json", "w") as f:
        json.dump(vocab, f)

    return vocab
