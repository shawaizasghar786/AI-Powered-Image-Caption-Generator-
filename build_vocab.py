import nltk
from collections import Counter
import json
def build_vocab(caption,threshold=5):
    counter=Counter()
    for caption in captions:
        tokens=nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {word: idx+1 for idx, word in enumerate(words)}
    vocab['<PAD>'] = 0
    vocab['<START>'] = len(vocab)
    vocab['<END>'] = len(vocab)+1
    vocab['<UNK>'] = len(vocab)+2

    with open("utils/vocab.json", "w")as f:
        json.dump(vocab,f)
    return vocab
    
