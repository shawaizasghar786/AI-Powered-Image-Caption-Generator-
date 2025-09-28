def generate_caption(image, encoder, decoder, vocab, max_len=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()
    feature = encoder(image.unsqueeze(0).to(device))
    caption = [vocab['<START>']]
    for _ in range(max_len):
        inputs = torch.tensor(caption).unsqueeze(0).to(device)
        output = decoder(feature, inputs)
        predicted = output.argmax(2)[:, -1].item()
        caption.append(predicted)
        if predicted == vocab['<END>']:
            break
    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([inv_vocab.get(idx, '') for idx in caption[1:-1]])
