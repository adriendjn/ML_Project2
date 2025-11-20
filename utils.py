def load_file(train_pos_file, train_neg_file, test_file):
    try : 
        with open(train_pos_file, 'r', encoding='utf-8') as f1:
            pos_tweets = [line.strip() for line in f1 if line.strip()]

    except FileNotFoundError:
        print(f" File {train_pos_file} not found")
        return []
    
    try : 
        with open(train_neg_file, 'r', encoding='utf-8') as f2:
            neg_tweets = [line.strip() for line in f2 if line.strip()]
    except FileNotFoundError:
        print(f" File {train_neg_file} not found")
        return []

    try :
        with open(test_file, 'r', encoding='utf-8') as f3:
            test_tweets = [line.strip() for line in f3 if line.strip()]
        
    except FileNotFoundError:
        print(f" File {test_file} not found")

    return pos_tweets, neg_tweets, test_tweets