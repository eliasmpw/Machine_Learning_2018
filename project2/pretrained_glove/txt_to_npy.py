import numpy as np
import pickle




def main():
    filename = 'glove.twitter.27B.25d.txt'
    dimensions = 25

    with open('../vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    vocab_reduced = dict()
    embeddings = []
    index = 0
    with open(filename) as f:
        for line in f:
            itervalues = iter(line.strip().split())
            word = next(itervalues)
            if word in vocab:
                vocab_reduced[word] = index
                index += 1
                vec = []
                for value in itervalues:
                    embeddings.append(float(value))
 
    embeddings = np.asarray(embeddings).reshape((-1, dimensions))
    np.save('embeddings' + str(dimensions), embeddings)

    with open('vocab_pretrained.pkl', 'wb') as f:
        pickle.dump(vocab_reduced, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
