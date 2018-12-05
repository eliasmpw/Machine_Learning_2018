import torch
from torch import nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 128
        filter_size = 3
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        #self.embeddings.weight.requires_grad=False
        self.conv = torch.nn.Conv2d(1, n_channels, kernel_size=(filter_size, embedding_dim))
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        dropout_prob = 0.5
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = F.dropout(x, dropout_prob)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred
