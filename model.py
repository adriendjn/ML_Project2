import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter


def load_file(pos_file,neg_file):

    try : 
        with open(pos_file, 'r', encoding='utf-8') as f1:
            pos_tweets = [line.strip() for line in f1 if line.strip()]

    except FileNotFoundError:
        print(f" Fichier {pos_file} non trouvé")
        return []
    
    try : 
        with open(neg_file, 'r', encoding='utf-8') as f2:
            neg_tweets = [line.strip() for line in f2 if line.strip()]
    except FileNotFoundError:
        print(f" Fichier {neg_file} non trouvé")
        return []
    return pos_tweets, neg_tweets
    

def build_vocabulary(tweets, min_freq=5):
    """Construire le vocabulaire avec les mots apparaissant au moins min_freq fois"""
    # Compter tous les mots
    word_counter = Counter()
    
    for tweet in tweets:
        words = tweet.split()
        word_counter.update(words)
    
    # Filtrer les mots avec au moins min_freq occurrences
    vocabulary = {word for word, count in word_counter.items() if count >= min_freq}
    
    print(f"📚 Vocabulaire construit:")
    print(f"   - Mots uniques total: {len(word_counter)}")
    print(f"   - Mots avec freq >= {min_freq}: {len(vocabulary)}")
    
    # Afficher les mots les plus fréquents
    print(f"   - 10 mots les plus fréquents:")
    for word, count in word_counter.most_common(10):
        print(f"      '{word}': {count} occurrences")
    
    return vocabulary, word_counter

tweet_pos, tweet_neg = load_file("twitter-datasets/train_pos.txt","twitter-datasets/train_neg.txt")
vocabulary_pos, word_counts_pos = build_vocabulary(tweet_pos, min_freq=5)
vocabulary_neg, word_counts_neg = build_vocabulary(tweet_neg, min_freq=5)

    
    


