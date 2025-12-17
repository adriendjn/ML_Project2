# Project Text Sentiment Classification

Welcome to the repository for our group for the text sentiment classification project, our group is composed of Rayan Ouidir, Grégory Preisig and Adrien Dejean. We will present briefly the content of the repository and how to build submissions for every scripts provided including our fined tuned BERTweet model providing 0.899/0.900 scores


## Baselines

As baselines we used the Glove template offered with the base files of the project and added one merging the features of TFIDF and Glove to be able to compare our final model to some others and assess its performances.

## Generating Word Embeddings: 

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```


Now given the co-occurrence matrix and the vocabulary, it is not hard to train GloVe word embeddings, that is to compute an embedding vector for wach word in the vocabulary. We suggest to implement SGD updates to train the matrix factorization, as in

```glove_solution.py```

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets `pos_train_full.txt`, `neg_train_full.txt`

## Building submissions for the baselines

In order to run the scripts to build the predictions of the baseline, the scripts are made to be run from within the the folder `glove` or `tfidf-glove` and once you are inside of it run respectively `python model_glove.py` or `python model_tfidf_glove.py`

## Building the final submission

Initially we built a jupyter notebook to take advantages of the possibility of using powerful GPUs from within google collab, the following notebook can be found under `BERT/modelBERT.ipynb` it basically works out the box but need the user to upload the zipped dataset to its google drive under `machineLearning/twitter-datasets.zip` and then you just need to select the runtime you need/is available.

## Running the run.py

In order to run locally run.py you need to install some packages for the virtual environment to function correctly, there is a file `requirement.txt` that can be found at the root of the repository that define the dependecies, then you just run the script by typing `python run.py` in the terminal which should run the script, note that the script is very slow as we initially designed it to be run on the GPUs we can use on google collab.