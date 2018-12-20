# Project Text Sentiment Classification

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

## Overview of the project's code
```
├── data                    <-Empty folder, copy .txt files here
├── Report
│   ├── Machine_Learning_AEJ_Sentiment.pdf    <-Report in .pdf format
│   └── Machine_Learning_AEJ_Sentiment.tex    <-Report in LaTeX format
├── build_vocab.sh   <-Creates a vocab.txt vocabulary from the .txt files
├── cooc.py          <-Coocurrence from a .pickle file
├── cut_vocab.sh     <-Cuts from a vocab.txt vocabulary file into a vocab_cut.txt
├── pickle_vocab.py  <-Converts from a vocab_cut.txt file to a vocab.pkl file
├── helpers.py       <-General custom helper methods
├── logreg.py        <-Contains the logistic regression methods
├── README.md        <-The README file of the project
````

## Prerequisites
1. You will need to have Anaconda installed, the following tutorials will guide you through the installation.

    Operating System | Tutorial
    --- | ---
    Mac | https://docs.anaconda.com/anaconda/install/mac-os/
    Windows | https://docs.anaconda.com/anaconda/install/windows/
    Ubuntu | https://docs.anaconda.com/anaconda/install/linux/

2. Once you have it installed, try running the following command on a terminal:

    ```
    python
    ```

    You should see a message similar to the following:
    ```
    Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    ```

3. For language processing corpora and lexical resources, we need to install **NLTK** (Natural Language Toolkit), by running the commands:
    ```
    pip install -U nltk
    ```

4. For the preprocessing steps, the **TextBlob** library is used. It is installed running the followind commands:
    ```
    pip install -U textblob
    python -m textblob.download_corpora
    ```

5. You will also need to have pytorch installed, please follow the instructions found in the pytorch web page: [Install Pytorch](https://pytorch.org/get-started/locally)

## Installing

1. In a terminal, change your directory to the location of the compressed zip file of the project.

    ```
    cd {path_of_zip_file}
    ```

2. Unzip the zip file of the project.
    ```
    unzip -a mlProject2.zip
    ```

3. Download the provided datasets `twitter-datasets.zip` from [here](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files), and extract the *.txt files from the zip.

4. Copy the downloaded files inside the **/data** folder.

## Running the prediction

1. If your terminal is not in the location of the project files, change your directory to that location.
    ```
    cd {path_of_project_files}
    ```

### Classification using Word-Vectors

<!-- For building a good text classifier, it is crucial to find a good feature representation of the input text. Here we will start by using the word vectors (word embeddings) of each word in the given tweet. For simplicity of a first baseline, we will construct the feature representation of the entire text by simply averaging the word vectors.

Below is a solution pipeline with an evaluation step:

### Generating Word Embeddings: 

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```

Now given the co-occurrence matrix and the vocabulary, it is not hard to train GloVe word embeddings, that is to compute an embedding vector for wach word in the vocabulary. We suggest to implement SGD updates to train the matrix factorization, as in

```glove_solution.py```

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets `pos_train_full.txt`, `neg_train_full.txt` -->

### Building a Text Classifier:

<!-- 1. Construct Features for the Training Texts: Load the training tweets and the built GloVe word embeddings. Using the word embeddings, construct a feature representation of each training tweet (by averaging the word vectors over all words of the tweet).

2. Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed features, using the scikit learn library, or your own code from the earlier labs. Recall that the labels indicate if a tweet used to contain a :) or :( smiley.

3. Prediction: Predict labels for all tweets in the test set. -->

4. Submission / Evaluation: A new file called "submit.csv" will be generated, which contains the predictions and can be uploaded to [crowdAI](https://www.crowdai.org/challenges/epfl-ml-text-classification/submissions), where you can verify the obtained classification score. 

## Extensions:
Naturally, there are many ways to improve your solution, both in terms of accuracy and computation speed. More advanced techniques can be found in the recent literature.
