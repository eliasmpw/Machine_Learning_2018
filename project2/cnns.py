import torch
import torch.nn.functional as F


class SimpleConvNet(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 64
        filter_size = 3
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv = torch.nn.Conv2d(1, n_channels, kernel_size=(filter_size, embedding_dim))
        self.fc = torch.nn.Linear(n_channels, 1)

    def get_criterion(self):
        return torch.nn.BCELoss(), torch.float

    def forward(self, x):
        dropout_prob = 0
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = F.dropout(x, dropout_prob)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred



class ClassificationNet(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 64
        filter_size = 3
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv = torch.nn.Conv2d(1, n_channels, kernel_size=(filter_size, embedding_dim))
        self.fc = torch.nn.Linear(n_channels, 2)

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss(), torch.long

    def forward(self, x):
        dropout_prob = 0
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = F.dropout(x, dropout_prob)
        x = self.fc(x)
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        _, pred = torch.max(F.softmax(logits, dim=1), 1)
        return pred



class NGrams(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 64
        kernel_sizes = [2, 3, 5]
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv1 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[0], embedding_dim))
        self.conv2 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[1], embedding_dim))
        self.conv3 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[2], embedding_dim))
        self.fc = torch.nn.Linear(3 * n_channels, 1)

    def get_criterion(self):
        return torch.nn.BCELoss(), torch.float

    def forward(self, x):
        dropout_prob = 0
 
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
        x = torch.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred



class DeepNGrams(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        n_channels = 64
        kernel_sizes = [2, 3, 5]
        linear_sizes = [3 * n_channels, n_channels, 16]
        
        embedding_dim = embeddings.shape[1]        
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv1 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[0], embedding_dim))
        self.conv2 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[1], embedding_dim))
        self.conv3 = torch.nn.Conv2d(1, n_channels, kernel_size=(kernel_sizes[2], embedding_dim))
        self.fc1 = torch.nn.Linear(linear_sizes[0], linear_sizes[1])
        self.fc2 = torch.nn.Linear(linear_sizes[1], linear_sizes[2])
        self.fc3 = torch.nn.Linear(linear_sizes[2], 1)

    def get_criterion(self):
        return torch.nn.BCELoss(), torch.float

    def forward(self, x):
        dropout_prob = 0
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        conved_1 = F.relu(self.conv1(x).squeeze(3))
        conved_2 = F.relu(self.conv2(x).squeeze(3))
        conved_3 = F.relu(self.conv3(x).squeeze(3))
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        x = F.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1), dropout_prob)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred



class ComplexConvNet(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        embedding_dim = embeddings.shape[1]  
        n_channels = [64, 64, 64]
        kernel_size = [(2, embedding_dim), (2, n_channels[0]), (2, n_channels[1])] 
                    
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv1 = torch.nn.Conv2d(1, n_channels[0], kernel_size=kernel_size[0])
        self.conv2 = torch.nn.Conv2d(1, n_channels[1], kernel_size=kernel_size[1])
        self.conv3 = torch.nn.Conv2d(1, n_channels[2], kernel_size=kernel_size[2])
        self.fc = torch.nn.Linear(n_channels[2], 1)

    def get_criterion(self):
        return torch.nn.BCELoss(), torch.float

    def forward(self, x):
        dropout_prob = 0
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = x.squeeze(3)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = F.relu(self.conv2(x))
        x = x.squeeze(3)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = F.relu(self.conv3(x))
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = F.dropout(x, dropout_prob)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred


class MultipleConvNet(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        
        embedding_dim = embeddings.shape[1]  
        n_channels = [64, 64, 64]
        kernel_size = [(2, embedding_dim), (2, 1), (2, 1)] 
                    
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad=False
        self.conv1 = torch.nn.Conv2d(1, n_channels[0], kernel_size=kernel_size[0])
        self.conv2 = torch.nn.Conv2d(1, n_channels[1], kernel_size=kernel_size[1])
        self.conv3 = torch.nn.Conv2d(1, n_channels[2], kernel_size=kernel_size[2])
        self.fc = torch.nn.Linear(n_channels[2], 1)

    def get_criterion(self):
        return torch.nn.BCELoss(), torch.float

    def forward(self, x):
        dropout_prob = 0
 
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.shape[2])
        x = x.unsqueeze(1)
        x = F.relu(self.conv2(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.shape[2])
        x = x.unsqueeze(1)
        x = F.relu(self.conv3(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = F.dropout(x, dropout_prob)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)
    
    def predict(self, x):
        pred = torch.round(self.forward(x))
        return pred
