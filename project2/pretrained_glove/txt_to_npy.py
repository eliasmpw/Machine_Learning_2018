def main():
    filename = 'glove.twitter.27B.50d.txt'
    


def pickle_vocab(vocab_name):
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab.pkl', 'wb') as f:
        filehandler = open(vocab_name,"wb")
        pickle.dump(vocab, f, filehandler)


if __name__ == '__main__':
    main()
