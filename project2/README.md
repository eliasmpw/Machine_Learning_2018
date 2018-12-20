# Project Text Sentiment Classification

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

## Overview of the project's code

```
├── data                    <-Empty folder, copy .txt files here
├── embeddings              <-Empty folder, copy vocabulary and embedding files here
├── Report
│   ├── Machine_Learning_AEJ_Sentiment.pdf    <-Report in .pdf format
│   └── Machine_Learning_AEJ_Sentiment.tex    <-Report in LaTeX format
├── build_vocab.sh          <-Creates a vocab.txt vocabulary from the .txt files
├── cnn_gpu.py              <-Run the Convolutional neural network using GPU
├── cnns.py                 <-Contains the Convolutional neuronal network methods
├── cooc_memory_friendly.py <-Memory friendly extraction of coocurrence from a .pickle file
├── cooc.py                 <-Coocurrence from a .pickle file
├── cut_vocab.sh            <-Cuts from a vocab.txt vocabulary file into a vocab_cut.txt
├── glove_solution.py         <-Generates embeddings from a vocabulary
├── helpers.py              <-General custom helper methods
├── logreg.py               <-Contains the logistic regression methods
├── pickle_vocab.py         <-Converts from a vocab_cut.txt file to a vocab.pkl file
├── pre_processing.ipynb    <-Runs the pre-processing steps in the tweets inside the *.txt data files
├── pre_processing.py       <-Helper methods for the pre-processing
├── run.py                  <-Runs the Convolutional neural network
├── README.md               <-The README file of the project
```

## Prerequisites

1. You will need to have Anaconda installed, the following tutorials will guide you through the installation.

    | Operating System | Tutorial                                            |
    | ---------------- | --------------------------------------------------- |
    | Mac              | https://docs.anaconda.com/anaconda/install/mac-os/  |
    | Windows          | https://docs.anaconda.com/anaconda/install/windows/ |
    | Ubuntu           | https://docs.anaconda.com/anaconda/install/linux/   |

2. Once you have it installed, try running the following command on a terminal:

    ```bash
    python
    ```

    You should see a message similar to the following:

    ```
    Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    ```

3. For language processing corpora and lexical resources, we need to install **NLTK** (Natural Language Toolkit), by running the commands:

    ```bash
    pip install -U nltk
    ```

4. For the preprocessing steps, the **TextBlob** library is used. It is installed running the followind commands:

    ```bash
    pip install -U textblob
    python -m textblob.download_corpora
    ```

5. You will also need to have PyTorch installed, since we are using this ML library for the base neural networks functionality, and we are building on top of it. To install PyTorch please follow the instructions found in their web page: [Install Pytorch](https://pytorch.org/get-started/locally)

## Installing

1. In a terminal, change your directory to the location of the compressed zip file of the project.

    ```bash
    cd {path_of_zip_file}
    ```

2. Unzip the zip file of the project.

    ```bash
    unzip -a mlProject2.zip
    ```

3. Download the provided datasets `twitter-datasets.zip` from [here](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files), and extract the `*.txt` files from the zip.

    ```bash
    unzip -a twitter-datasets.zip
    ```

4. Move the downloaded `*.txt` files inside the **/data** folder.

5. Download the pretrained vocabulary from [here](https://drive.google.com/open?id=1UCnuDxEICn8SBGxlEet7M8NCy32sLYvP), and the pretrained embeddings from [here](https://drive.google.com/open?id=14K8JSlsugyw2fIP4IbHoyKZWUHnonhT5).

6. Move the downloaded `vocab_pretrained.pkl` and `embeddings200_pretrained.npy` files inside the **/embeddings** folder.

## Text Classifier - Running the prediction

1. If your terminal is not in the location of the project files, change your directory to that location.

    ```bash
    cd {path_of_project_files}
    ```

2. Run the run.py script in the terminal.
    ```bash
    python run.py
    ```
    A new file called **"submissionX.csv"** will be generated, which contains the predictions.

3) The **"submissionX.csv"** file can be uploaded to [crowdAI](https://www.crowdai.org/challenges/epfl-ml-text-classification/submissions), where you can verify the obtained classification score.
   If you used the `vocab_pretrained.pkl` and `embeddings200_pretrained.npy` files you will reproduce the best submission made by this group.

## Pre-processing

In order to pre-process the tweets, you should edit and run the **pre_processing.ipynb** jupyter notebook, please follow these instructions:

1. Start Jupyter Notebook

    Run this command:

    ```bash
    jupyter notebook
    ```

    This will launch a browser window, with a file explorer open.

2. Open notebook

    Navigate through the folders, and open the **pre_processing.ipynb** file.

3. Set Feature treatment methods

    Inside the second cell you can modify the following variables:

    - **already_cleaned_neg** = True/False

    - **already_cleaned_pos** = True/False

    - **already_cleaned_neg_full** = True/False

    - **already_cleaned_pos_full** = True/False

    - **already_cleaned_test** = True/False

    The previous variables will determine which of the files will be pre-processed. If you set them to False, and run the next cells in the notebook, they will be pre-processed again. The pre-processing will be skipped for the ones with a value of True.

    If you want to try different combinations of the pre-processing steps you can change the follwing variables:

    - **duplicates** = True
    - **emojis** = False
    - **punctuaction** = False
    - **handle_number** = False
    - **special_symbols** = True
    - **moreLetters** = False
    - **contractions** = False
    - **clean_stopwords** = False
    - **spelling** = False
    - **lemmatize** = False

    If you set them to True and run the next cells, the will be pre-processed depending on previous choise. Please keep in mind that if you set the spelling to true it will take a long time to finish the pre-processing (depending on the specifics of your machine, 6-8 hours).

4. Run the cells of the notebook

    Now open the **Cell** menu and click on the option **Run All**, or manually run every cell.

    You will see the results/outputs between cells, and in the `data/pre_processed/` path you will find the new pre-processed `.txt` and `.pkl` files.

## Generating Vocabulary and Embeddings:

If you want to build your own vocabulary and embeddings, follow these instructions:

NOTE: These scripts have the `.txt` test data filenames set to `train_pos_full.txt` and `train_neg_full.txt`. If you wish to generate embeddings from a smaller dataset, you should modify those filenames in the code.

### Generate Vocabulary

In order to load the training tweets and construct a vocabulary list of words appearing at least 5 times, run the following commands.

    ```bash
    build_vocab.sh
    cut_vocab.sh
    python pickle_vocab.py
    ```

This will output a `vocab.pkl` vocabulary file.

### Generate GloVe Embeddings

For this you will need either your own `.pkl` file or the preprocessed one available [here](https://drive.google.com/open?id=1UCnuDxEICn8SBGxlEet7M8NCy32sLYvP).

#### Own Glove Embeddings

You can train your GloVe word embeddings, that is to compute an embedding vector for each word in the vocabulary, by using the following commands.

NOTE: The `cooc.py` script provided to us with the project description is prone to crash, so we are using `cooc_memory_friendly.py` which is an optimized version of it. They should be interchangeable.
You might need to modify the script to have the correct filenames for the vocabulary and data.

    ```bash
    python cooc_memory_friendly.py
    python glove_solution.py
    ```

This will output a `embeddings*.npy` file.

### Using our vocabulary with Stanford's embeddings

For better results, we used an embedding matrix which was created at the [Standford University](https://nlp.stanford.edu/projects/glove/) and is based on 2,6 Billion tweets.
From this embedding matrix, we will extract the embeddings that we need for our vocabulary.

1. Download Standford's embedding matrix from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

2. Extract the content of the zip file
    ```bash
    unzip -a glove.twitter.27B.zip
    ```
3. Modify the `glove_txt_to_npy.py` so it points to the correct vocabulary file on line 11, then run the script:
    ```bash
    python glove_txt_to_npy.py
    ```

This will output a `embeddings*.npy` file.

## Authors

-   **Andres Ivan Montero Cassab**
-   **Jonas Florian Jäggi**
-   **Elias Manuel Poroma Wiri**
