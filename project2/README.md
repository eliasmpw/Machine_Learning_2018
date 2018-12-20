# Project Text Sentiment Classification

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

## Overview of the project's code

```
├── data                    <-Empty folder, copy .txt files here
├── Report
│   ├── Machine_Learning_AEJ_Sentiment.pdf    <-Report in .pdf format
│   └── Machine_Learning_AEJ_Sentiment.tex    <-Report in LaTeX format
├── build_vocab.sh   <-Creates a vocab.txt vocabulary from the .txt files
├── cnn_gpu.py          <-Run the Convolutional neural network using GPU
├── cnns.py          <-Contains the Convolutional neuronal network methods
├── cooc_memory_friendly.py          <-Memory friendly extraction of coocurrence from a .pickle file
├── cooc.py          <-Coocurrence from a .pickle file
├── cut_vocab.sh     <-Cuts from a vocab.txt vocabulary file into a vocab_cut.txt
├── glove_solution.py          <-Generates embeddings from a vocabulary
├── helpers.py       <-General custom helper methods
├── logreg.py        <-Contains the logistic regression methods
├── pickle_vocab.py  <-Converts from a vocab_cut.txt file to a vocab.pkl file
├── pre_processing.ipynb  <-Runs the pre-processing steps in the tweets inside the *.txt data files
├── pre_processing.py  <-Helper methods for the pre-processing
├── README.md        <-The README file of the project
```

## Prerequisites

1. You will need to have Anaconda installed, the following tutorials will guide you through the installation.

    | Operating System | Tutorial                                            |
    | ---------------- | --------------------------------------------------- |
    | Mac              | https://docs.anaconda.com/anaconda/install/mac-os/  |
    | Windows          | https://docs.anaconda.com/anaconda/install/windows/ |
    | Ubuntu           | https://docs.anaconda.com/anaconda/install/linux/   |

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

3. Download the provided datasets `twitter-datasets.zip` from [here](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files), and extract the \*.txt files from the zip.

4. Copy the downloaded files inside the **/data** folder.

## Text Classifier - Running the prediction

1. If your terminal is not in the location of the project files, change your directory to that location.

    ```
    cd {path_of_project_files}
    ```

2. Run the cnn_gpu.py script in the terminal.
    ```
    python cnn_gpu.py
    ```
    A new file called **"cnn.csv"** will be generated, which contains the predictions.

3) The **"cnn.csv"** file can be uploaded to [crowdAI](https://www.crowdai.org/challenges/epfl-ml-text-classification/submissions), where you can verify the obtained classification score.

## Pre-processing

In order to pre-process the tweets, you should edit and run the **pre_processing.ipynb** jupyter notebook, please follow these instructions:

1. Start Jupyter Notebook

    Run this command:

    ```
    jupyter notebook
    ```

    This will launch a browser window, with a file explorer open.

2. Open notebook

    Navigate through the folders, and open the **pre_processing.ipynb** file.

3. Set Feature treatment methods

    Inside the second cell you can modify the following values:

    - **already_cleaned_neg** = True/False

    - **already_cleaned_pos** = True/False

    - **already_cleaned_neg_full** = True/False

    - **already_cleaned_pos_full** = True/False

    - **already_cleaned_test** = True/False

    If you set them to False, and run the next cells in the notebook, they will be pre-processed again.
    The pre-processing will be skipped for the ones with a value of True.

6) Run the cells of the notebook

    Now open the **Cell** menu and click on the option **Run All**, or manually run every cell.

    You will see the results/outputs between cells, and in the `data/pre_processed/` path you will find the new pre-processed .txt files.

## Classification using Word-Vectors

If you want to build your own vocabulary and embeddings and vocabulary, follow these instructions:

### Generating Word Embeddings:

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```

Then you can to train GloVe word embeddings, that is to compute an embedding vector for wach word in the vocabulary.

`glove_solution.py`

## Authors

-   **Andres Ivan Montero Cassab**
-   **Jonas Florian Jäggi**
-   **Elias Manuel Poroma Wiri**
