import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models import EncoderCNN, DecoderRNN
from utils.dataset import FlickrDataset
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = FlickrDataset("data/Flickr8k_Dataset", "data/Flickr8k_text/Flickr8k.token.txt", "utils/vocab.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: zip(*x))

with open("utils/vocab.json") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

encoder = EncoderCNN(EMBED_SIZE).to(device)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for images, captions in dataloader:
        images = torch.stack(images).to(device)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab['<PAD>']).to(device)
        outputs = decoder(encoder(images), captions[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(encoder.state_dict(), f"models/encoder_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"models/decoder_epoch{epoch+1}.pth")
