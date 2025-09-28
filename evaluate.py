def beam_search_caption(encoder, decoder, image, vocab, beam_size=3, max_len=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image.unsqueeze(0).to(device))  # (1, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_out.size(-1))  # (1, num_pixels, encoder_dim)

    k = beam_size
    vocab_size = len(vocab)
    start_token = vocab['<START>']
    end_token = vocab['<END>']

    sequences = [[start_token]]
    scores = torch.zeros(k, 1).to(device)

    h = decoder.init_h(encoder_out.mean(dim=1))
    c = decoder.init_c(encoder_out.mean(dim=1))

    for _ in range(max_len):
        all_candidates = []
        for i in range(len(sequences)):
            seq = sequences[i]
            inputs = torch.tensor([seq[-1]]).to(device)
            embedding = decoder.embedding(inputs)
            context, _ = decoder.attention(encoder_out, h)
            lstm_input = torch.cat([embedding.squeeze(0), context.squeeze(0)], dim=0)
            h, c = decoder.lstm(lstm_input.unsqueeze(0), (h, c))
            output = decoder.fc(h)
            log_probs = torch.log_softmax(output, dim=1)
            top_k_probs, top_k_words = log_probs.topk(k)

            for j in range(k):
                candidate = seq + [top_k_words[0][j].item()]
                score = scores[i] + top_k_probs[0][j]
                all_candidates.append((candidate, score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = [seq for seq, score in ordered[:k]]
        scores = torch.stack([score for seq, score in ordered[:k]])

        if all(seq[-1] == end_token for seq in sequences):
            break

    best_seq = sequences[0]
    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([inv_vocab.get(idx, '') for idx in best_seq[1:-1]])
