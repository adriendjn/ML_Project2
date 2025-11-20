from collections import Counter
import numpy as np

#### Potentiel ajout d'une fonction de nettoyage des tweets 

def load_file(train_pos_file, train_neg_file, test_file):
    pos_tweets, neg_tweets, test_tweets = [], [], []
    try : 
        with open(train_pos_file, 'r', encoding='utf-8') as f1:
            pos_tweets = [line.strip() for line in f1 if line.strip()]

    except FileNotFoundError:
        print(f" File {train_pos_file} not found")
    
    try : 
        with open(train_neg_file, 'r', encoding='utf-8') as f2:
            neg_tweets = [line.strip() for line in f2 if line.strip()]
    except FileNotFoundError:
        print(f" File {train_neg_file} not found")

    try :
        with open(test_file, 'r', encoding='utf-8') as f3:
            test_tweets = [line.strip().split(',', 1)[-1] for line in f3 if line.strip()]
        
    except FileNotFoundError:
        print(f" File {test_file} not found")

    return pos_tweets, neg_tweets, test_tweets

# def 

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            if word:
                vocab[word] = idx
    return vocab

def build_vocabulary(tweets, min_freq=5):
 
    word_counter = Counter()
    
    for tweet in tweets:
        words = tweet.split()
        word_counter.update(words)
    
    vocabulary = {word for word, count in word_counter.items() if count >= min_freq}
    
    print(f"📚 Vocabulaire construit:")
    print(f"   - Mots uniques total: {len(word_counter)}")
    print(f"   - Mots avec freq >= {min_freq}: {len(vocabulary)}")
    return vocabulary, word_counter

def vec_tweet(tweet, vocab, embeddings):
    words = tweet.split()
    vecs = []

    for w in words:
        if w in vocab:
            idx = vocab[w]
            vecs.append(embeddings[idx])
    if vecs:
        return np.array(np.mean(vecs, axis=0))
    else: return np.array(np.zeros(embeddings.shape[1]))
    

def tweets_to_features(tweets, vocab, embeddings):
    x = []
    for tweet in tweets:
        x.append(vec_tweet(tweet, vocab, embeddings))
    return np.array(x)

def create_submission(prediction, filename):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for idx, pred in enumerate(prediction):
            if pred == 1:
                f.write(f"{idx},1\n")
            else: f.write(f"{idx},-1\n")