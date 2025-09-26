import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet=models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        self.linear=nn.Linear(resnet.fc.in_feature,embed_size)
        self.bn=nn.BatchNorm1d(embed_size)
    def forward(self,image):
        with torch.no_grad():
            features=self.resnet(image).squeeze()
            return self.bn(self.linear(features))

    class DecoderRNN(nn.Module):
      def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)   

    def forward(self,features,captions):
       embeddings=self.embed(captions)
       inputs=torch.cat((features.unsqueeze(1),embeddings),1)
       hiddens,_=self.lstm(inputs)
       outputs=self.linear(hiddens)
       return outputs
