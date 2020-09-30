import numpy as np
import pandas as pd
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import lightgbm

working_path = r'C:\Users\97254\Desktop\Real or Not NLP'

train = pd.read_csv(working_path + r'\train.csv')
test = pd.read_csv(working_path + r'\test.csv')
sub = pd.read_csv(working_path + r'\sample_submission.csv')

# - Pipeline: Sentence embedding -- train light gbm

# ------ Text Pre-Processing ------ #
# ---  bert - tokenaizer --- #


def text_preprocessing(df):

    # --- convert to lower letters --- #
    df['text'] = df['text'].apply(lambda row: row.lower())

    # --- remove numbers --- #
    df['text'] = df['text'].apply(lambda row: re.sub(r'\d+', '', row))

    # --- remove punctuation --- #
    df['text'] = df['text'].apply(lambda row: ''.join([char for char in row if char not in string.punctuation]))

    # --- Remove whitespaces --- #
    df['text'] = df['text'].apply(lambda row: row.strip())

    # --- create list of sentences --- #
    sentences = [sen for sen in df['text']]
    return sentences


def bert_tokenize(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences_tokenized = [tokenizer.tokenize(sen) for sen in df['text'].tolist()]
    sentences_tokenized = [' '.join(i) for i in sentences_tokenized]
    return sentences_tokenized


train_sentences = bert_tokenize(train)
test_sentences = bert_tokenize(test)
# ------ Sentence Embedding ------ #
# --- using bert - pre-trained model --- #


def bert_embedding(sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = np.asarray(sentence_embeddings)
    return sentence_embeddings


x_train = bert_embedding(train_sentences)
x_test = bert_embedding(test_sentences)
# ------ Modeling ------ #
y_train = train['target']
# --- Logistic Regression --- #
C = np.logspace(0, 4, num=10)
penalty = ['l1', 'l2']
solver = ['liblinear', 'saga']
lr_params = dict(C=C, penalty=penalty, solver=solver)
lr = LogisticRegression()
RSlr = RandomizedSearchCV(param_distributions=lr_params, estimator=lr, scoring="f1", cv=5, n_iter=10, refit=True, verbose=1)
RSlr.fit(X=x_train, y=y_train)

sub['target'] = RSlr.predict(X=x_test)
sub.to_csv(working_path + r'\sub.csv', index=False)

# light - gbm
lgb = lightgbm.LGBMClassifier()
lgb.fit(x_train, y_train)
sub['target'] = lgb.predict(X=x_test)
sub.to_csv(working_path + r'\sub.csv', index=False)