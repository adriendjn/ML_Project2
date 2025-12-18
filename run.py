import torch
from transformers import set_seed, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from utils import load_tweets, create_csv_submission, compute_metrics
import numpy as np
import random
# Fixing the seeds for repeatability
random.seed(42)
np.random.seed(42)
set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Declaration of the paths for the datasets
TRAIN_POS_FILE = "train_pos.txt"
TRAIN_NEG_FILE = "train_neg.txt"
TEST_DATA_FILE = "test_data.txt"
# Loading the datasets
tweet_pos, tweet_neg, tweet_test = load_tweets(TRAIN_POS_FILE, TRAIN_NEG_FILE, TEST_DATA_FILE)
# Making the arrays combining pos and neg sets
tweets = tweet_pos + tweet_neg
# Making labels with 1 for positive tweets and 0 for negative tweets
labels = [1]*len(tweet_pos) + [0]*len(tweet_neg)
# Creating a torch Dataset
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = dict(encodings) # Conversion en dict pour éviter le KeyError
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
# Splitting the data for split validation
train_tweets, eval_tweets, train_labels, eval_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42, stratify=labels)
# Declaration of the model name
model_name = "vinai/bertweet-base"
# Loading the tokenizer for bertweet
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizing train data
train_encodings = tokenizer(train_tweets, truncation=True, padding=True, max_length=128)

# Tokenizing test data
eval_encodings = tokenizer(eval_tweets, truncation=True, padding=True, max_length=128)
# Transforming the datasets to torch Datasets
train_dataset = TwitterDataset(train_encodings, train_labels)
eval_dataset = TwitterDataset(eval_encodings, eval_labels)
# Loading the pretrained BERTweet for finetuning
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# Declaration of the arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    lr_scheduler_type="cosine",
    learning_rate=1e-5,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)
# Declaration of the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
# Starting the training
trainer.train()
# Tokenizing the test set and transforming it to torch Dataset
final_encoding = tokenizer(tweet_test, truncation=True, padding=True, max_length=128)
final_dataset = TwitterDataset(final_encoding, [0]*len(tweet_test))
# Doing the final prediction
predictions = trainer.predict(final_dataset)
final_proba_predictions = predictions.predictions.argmax(-1)
# Making the 1 for positive prediction and -1 for negative prediction for aicrowd compliance
final_predictions = [1 if i == 1 else -1 for i in final_proba_predictions]
# Making the submission.csv file
create_csv_submission(range(1, len(final_predictions)+1), final_predictions, "submissionBERT.csv")