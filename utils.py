from collections import Counter
from os import path
import numpy as np
import csv

#### Potentiel ajout d'une fonction de nettoyage des tweets 

def load_txt_file(file_path: str):
    """Load a text file's content from a relative path.

    Load text data using UTF-8 encoding, from a file located at the relative path ``file_path``.
    Strip leading and trailing whitespaces if any and discard blank lines.

    Parameters
    ----------
    file_path: str
        The text file relative path from working directory
    
    Returns
    -------
        file_data: list[str]
            A list of strings corresponding to the loaded file's lines, excluding blank or whitespace only lines.

    Raises
    ------
        Error: FileNotFoundError
            An error occured when opening the file.
    """
    fp = path.join(path.split(__file__)[0], path.normcase(file_path))
    try: 
        with open(fp, 'rt', encoding='utf-8') as f:
            file_data = [line.strip() for line in f if line.strip()]

    except FileNotFoundError:
        print(f" File {fp} not found")
        raise 
    return file_data

def load_tweets(train_pos_file, train_neg_file, test_file):
    """Load positive, negative and test tweets from the relative file paths ``train_pos_file``, ``train_neg_file`` and ``test_file``.

    Parameters
    ----------
    train_pos_file: str
        The positive tweets relative file path.
    train_neg_file: str
        The negative tweets relative file path.
    test_file: str
        The test tweets relative file path.
    
    Returns
    -------
    pos_tweets, neg_tweets, test_tweets: tuple[list[str], list[str], list[str]]
        A tuple of three lists of strings containing the lines of each file.
    """
    pos_tweets, neg_tweets, test_tweets = [], [], []
    try : 
        TWITTER_DATASET_PATH = "./twitter-datasets/"
        pos_tweets = load_txt_file(TWITTER_DATASET_PATH + train_pos_file)
        neg_tweets = load_txt_file(TWITTER_DATASET_PATH + train_neg_file)
        test_tweets = load_txt_file(TWITTER_DATASET_PATH + test_file)
    except FileNotFoundError:
        print("Error with one or more files, empty data returned")

    return pos_tweets, neg_tweets, test_tweets

def load_vocab(vocab_file):
    """Load vocabulary words from the relative file path ``vocab_file``

    Parameters
    ----------
    vocab_file: str
        The vocabulary relative file path.

    Returns
    -------
    vocab: dict[str, int]
        A dictionnary of words with their corresponding line number in the file.
    """
    vocab = {line : idx for idx, line in enumerate(load_txt_file(vocab_file))}
    return vocab

def build_vocabulary(tweets, min_freq=5):
    """Build a word vocabulary from a list of strings, optionally restricted to words with a minimum frequency of ``min_freq``.

    Parameters
    ----------
    tweets: list of str
        A list of tweets from which to extract words.
    min-freq: int, default=5
        minimum frequency a word has to have to be included in the vocabulary.

    Returns
    -------
    vocabulary, word_counter: tuple[set, Counter]
        Set of words included in the vocabulary, Counter object mapping each word in the tweets to it's frequency.
    """

    word_counter = Counter()
    
    for tweet in tweets:
        words = tweet.split()
        word_counter.update(words)
    
    vocabulary = {word for word, count in word_counter.items() if count >= min_freq}
    
    print(f"Vocabulary constructed:")
    print(f"    Total unique words: {len(word_counter)}")
    print(f"    Words with frequency >= {min_freq}: {len(vocabulary)}")
    return vocabulary, word_counter

def vec_tweet(tweet, vocab, embeddings):
    """Build the embedding vector for a tweet from each of its word.

    Parameters
    ----------
    tweet: str
        A tweet as a string.
    vocab: dict str to int
        The vocabulary mapping each word to its frequency.
    embeddings: ndarray of float64
        The embeddings for each word in the vocabulary.

    Returns
    -------
    ndarray of float64
    """
    vecs = [embeddings[vocab[w]] for w in tweet.split() if w in vocab]

    if vecs:
        return np.array(np.mean(vecs, axis=0))
    else: return np.array(np.zeros(embeddings.shape[1]))
    

def tweets_to_features(tweets, vocab, embeddings):
    """Convert a list of tweets to a normalized feature representation.

    Parameters
    ----------
    tweets: list of str
        List of tweets to convert.
    vocab: dict str to int
        The vocabulary mapping each word to its frequency.
    embeddings: ndarray of float64
        The embeddings for each word in the vocabulary.

    Returns
    -------
    ndarray of float64
    """
    x = [vec_tweet(tweet, vocab, embeddings) for tweet in tweets]
    return np.array(x)

def create_csv_submission(ids, y_pred, file_name):
    """
    This function creates a csv file named `file_name` in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with `ids` and the second with `y_pred`.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(file_name, "w") as csv_file:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csv_file, delimiter = ",", fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(ids)):
            writer.writerow({"Id": int(ids[i]), "Prediction": int(y_pred[i])})

def compute_metrics(pred):
    labels = pred.label_ids
    acc = accuracy_score(labels, pred.predictions.argmax(-1))
    f1 = f1_score(labels, pred.predictions.argmax(-1))
    return {"accuracy": acc, "f1": f1}