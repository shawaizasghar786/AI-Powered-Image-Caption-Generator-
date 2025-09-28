import torch
from models import EncoderCNN, DecoderWithAttention
from PIL import Image
import torchvision.transforms as transforms
import json

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def generate_caption(image, encoder, decoder, vocab, max_len=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image.unsqueeze(0).to(device))
    encoder_out = encoder_out.view(1, -1, encoder_out.size(-1))

    h, c = decoder.init_hidden_state(encoder_out)
    word = torch.tensor([vocab['<START>']]).to(device)
    caption = [vocab['<START>']]

    for _ in range(max_len):
        embedding = decoder.embedding(word)
        context, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context
        lstm_input = torch.cat([embedding.squeeze(0), context], dim=1)
        h, c = decoder.decode_step(lstm_input, (h, c))
        output = decoder.fc(h)
        predicted = output.argmax(1).item()
        caption.append(predicted)
        word = torch.tensor([predicted]).to(device)
        if predicted == vocab['<END>']:
            break

    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([inv_vocab.get(idx, '') for idx in caption[1:-1]])

# Example usage
if __name__ == "__main__":
    with open("utils/vocab.json") as f:
        vocab = json.load(f)

    encoder = EncoderCNN()
    decoder = DecoderWithAttention(EMBED_SIZE, HIDDEN_SIZE, len(vocab))
    encoder.load_state_dict(torch.load("models/encoder_epoch20.pth", map_location='cpu'))
    decoder.load_state_dict(torch.load("models/decoder_epoch20.pth", map_location='cpu'))

    image = load_image("data/Flickr8k_Dataset/123456789.jpg")
    caption = generate_caption(image, encoder, decoder, vocab)
    print("Generated Caption:", caption)
