import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models import EncoderCNN, DecoderWithAttention
from utils.dataset import FlickrDataset
from config import *
import json
from tqdm import tqdm

image_dir = "Flickr8k_Dataset"
caption_file = "Flickr8k_text/Flickr8k.token.txt"
vocab_path = "utils/vocab.json"  # or just "vocab.json" if it's in root


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and vocab
dataset = FlickrDataset("data/Flickr8k_Dataset", "data/Flickr8k_text/Flickr8k.token.txt", "utils/vocab.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: zip(*x))

with open("utils/vocab.json") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# Initialize models
encoder = EncoderCNN().to(device)
decoder = DecoderWithAttention(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    encoder.train()
    decoder.train()
    total_loss = 0

    for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images = torch.stack(images).to(device)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab['<PAD>']).to(device)

        encoder_out = encoder(images)
        outputs = decoder(encoder_out, captions[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    torch.save(encoder.state_dict(), f"models/encoder_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"models/decoder_epoch{epoch+1}.pth")
