import torch
import torch.nn as nn
import torchvision.models as models

# Encoder CNN using ResNet50
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super().__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 14, 14)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size, num_pixels, 2048)
        return features

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))  # (batch_size, num_pixels, 1)
        alpha = self.softmax(att.squeeze(2))  # (batch_size, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return context, alpha

# Decoder with Attention
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, attention_dim=256):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        caption_len = captions.size(1)

        embeddings = self.embedding(captions)  # (batch_size, caption_len, embed_size)
        h, c = self.init_hidden_state(encoder_out)

        outputs = torch.zeros(batch_size, caption_len, vocab_size).to(encoder_out.device)

        for t in range(caption_len):
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar
            context = gate * context
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            outputs[:, t] = output

        return outputs
