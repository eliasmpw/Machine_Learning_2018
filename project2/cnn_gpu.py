import numpy as np
import pickle
import time

import torch
from torch import optim
from torch.autograd import Variable
import torch.utils.data as utils

from helpers import *
from cnns import *


# define file paths
path_embeddings = 'pretrained_glove/embeddings200_pretrained_reduced.npy'
path_vocab = 'pretrained_glove/vocab_pretrained_reduced.pkl'
path_train_pos = 'pos_train.txt'
path_train_neg = 'neg_train.txt'
path_test = 'test_data.txt'

# define file names
submission_file = 'cnn1_reduced_32.csv'
network_file = 'network.pt'


# load word embeddings
embeddings = np.load(path_embeddings)
# add line of zeroes to the embeddings for empty words
embeddings = np.append(np.zeros((1, embeddings.shape[1])), embeddings, axis=0)
# load vocabulary
with open(path_vocab, 'rb') as f:
    vocab = pickle.load(f)


# find maximal tweet length (number of words)
longest = 0
for file in [path_train_pos, path_train_neg, path_test]:
    with open(file) as f:
        for line in f:
            length = len(line.strip().split())
            if length > longest:
                longest = length          
print("Longest tweet has {:d} words".format(longest))


'''
For each tweet we create an array containing the indexes of the words in
the word embedding matrix. All of these vectors have equal length equivalent
to the number of words in the longest tweet. If a tweet doesn't fill everything,
we pad with 0. this index corresponds to the embedding [0, 0, 0, ... 0, 0].

Data type is int32 in order to minimize memory usage.

For the labels, we use the value 0 for negative tweets and 1 for positive tweets.
'''

#initiate empty feature and label lists
x = []
y = []

# process positive tweets
with open(path_train_pos) as f:
    for line in f:
        tweet = np.int32(np.zeros((longest)))
        wordcount = 0
        y.append(1)
        for word in line.strip().split():
            index = vocab.get(word, -1);
            # skip words for which we have no embedding
            if(index != -1):
                tweet[wordcount] = index + 1
                wordcount += 1
        x.append(tweet)

# process negative tweets
with open(path_train_neg) as f:
    for line in f:
        tweet = np.int32(np.zeros((longest)))
        wordcount = 0
        y.append(0)
        for word in line.strip().split():
            index = vocab.get(word, -1);
            # skip words for which we have no embedding
            if(index != -1):
                tweet[wordcount] = index + 1
                wordcount += 1
        x.append(tweet)

# transform to numpy array
x_train = np.asarray(x)
y_train = np.asarray(y)
 
# Shuffle tweets
x_train, y_train = shuffle(x_train, y_train)


# We proceed process test tweets in the same way as the training tweets
x = []

with open(path_test) as f:
    for line in f:
        tweet = np.int32(np.zeros((longest)))
        wordcount = 0
        for word in line.strip().split():
            index = vocab.get(word, -1);
            # skip words for which we have no embedding
            if(index != -1):
                tweet[wordcount] = index + 1
                wordcount += 1
        x.append(tweet)

        # convert to numpy array
x_test = np.asarray(x)


# specify the network you want to use
net = SimpleConvNet(torch.from_numpy(embeddings).float()).cuda()
# decide how many training tweets to use for validation
val_prop = 10000
batch_size = 1024
epochs = 5
# defines after how many batches loss and accuracy are displayed
print_every = 20
# choose loss function
criterion = torch.nn.BCELoss()
# choose optimizer (Adam does fine most of the time)
optimizer = torch.optim.Adam(net.parameters())


# cut validation data from training data.
# convert numpy arrays to torch tensors
x_val_torch = torch.from_numpy(x_train[0:val_prop, :])
y_val_torch = torch.from_numpy(y_train[0:val_prop]).float()
x_train_torch = torch.from_numpy(x_train[val_prop + 1:, :])
y_train_torch = torch.from_numpy(y_train[val_prop + 1:]).float()

# create batch loaders
train_set = utils.TensorDataset(x_train_torch, y_train_torch)
train_loader = utils.DataLoader(train_set, batch_size, shuffle=False)
val_set = utils.TensorDataset(x_val_torch, y_val_torch)
val_loader = utils.DataLoader(val_set, batch_size, shuffle=False)

# Run training
steps = 0
running_loss = 0
start = time.time()

for e in range(epochs):
    for tweets, labels in iter(train_loader):
        steps += 1
        # converting to Variable is necessary in order to compute the gradient later
        inputs = Variable(tweets).cuda()
        targets = Variable(labels).cuda()
        # set gradient to zero
        optimizer.zero_grad()
        # forward inputs through the net
        outputs = net.forward(inputs.long())
        # compute loss and gradient
        loss = criterion(outputs, targets)
        loss.backward()
        # update weights
        optimizer.step()
        running_loss += loss.data[0]
        
        if steps % print_every == 0:
            stop = time.time()
            accuracy = 0
            n = 0
            # compute accuracy on validation set
            net.eval()
            for tweets, labels in iter(val_loader):
                predictions = net.predict(tweets.long().cuda()).cpu()
                accuracy += sum(predictions.data.numpy() == labels.data.numpy())
                n += labels.data.numpy().size
            net.train()            

            print("Epoch {} / {}\t".format(e+1, epochs),
                  "Loss {:.4f}\t".format(running_loss / print_every),
                  "Validation accuracy {:.4f}\t".format(accuracy / n),
                  "{:.4f} s/batch".format((stop - start)/print_every))
            running_loss = 0
            start = time.time()


# Save the trained network
torch.save(net.cpu(), network_file)

# Load a saved network
net = torch.load(network_file).cuda()


# compute accuracy based on a part of the training data
net.eval()
for tweets, labels in iter(val_loader):
    predictions = net.predict(tweets.long().cuda()).cpu()
    accuracy += sum(predictions.data.numpy() == labels.data.numpy())
    n += labels.data.numpy().size
            
print("Accuracy on validation set: {:.4f}".format(accuracy / n))


# Compute preditions and transform the labels to (-1, 1)
net.eval()
test_loader = utils.DataLoader(torch.from_numpy(x_test), batch_size, shuffle = False)
submission_labels = np.zeros((0))
for tweets in iter(test_loader):
    predictions = net.predict(tweets.long().cuda()).cpu()
    labels = predictions.data.numpy() * 2 - 1
    submission_labels = np.concatenate((submission_labels, labels), axis=0)
    
# save predicitons on the test set to csv file.
ids = np.arange(len(submission_labels)) + 1
create_csv_submission(ids, submission_labels, submission_file)
print("Submissions saved as", submission_file)





