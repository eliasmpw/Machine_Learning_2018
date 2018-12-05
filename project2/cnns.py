import torch
from torch import nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 256
        filter_size = 3
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
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


class NGrams(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 128
        kernel_sizes = [2, 3, 5]
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv1 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[0], embedding_dim))
        self.conv2 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[1], embedding_dim))
        self.conv3 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[2], embedding_dim))
        self.fc = nn.Linear(3 * n_channels, 1)

    def forward(self, x):
        dropout_prob = 0.5
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        conved_1 = F.relu(self.conv1(x).squeeze(3))
        conved_2 = F.relu(self.conv2(x).squeeze(3))
        conved_3 = F.relu(self.conv3(x).squeeze(3))
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        x = F.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1), dropout_prob)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred
