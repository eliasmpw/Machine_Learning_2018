import numpy as np

def main():
    filename = 'glove.twitter.27B.50d.txt'
    
    embeddings = np.zeros((0, 50))
    with open(filename) as f:
        for line in f:
            itervalues = iter(line.strip().split())
            next(itervalues)
            vec = np.asarray([])
            for word in itervalues:
                vec.append(float(word))
            embeddings.append(vec, axis=0)
    embeddings = np.asarray(embeddings)
    print(embeddings.shape)
    print(embeddings[1:3, :])



'''
def pickle_vocab(vocab_name):
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab.pkl', 'wb') as f:
        filehandler = open(vocab_name,"wb")
        pickle.dump(vocab, f, filehandler)
****
'''

if __name__ == '__main__':
    main()
