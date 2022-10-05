# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import glob
import matplotlib.pyplot as plt
import spacy
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
# -

# ### Read Spam Emails:

# Some of the documents in our dataset are encoded in ISO-8859, which will cause issues as we try to process the data. We'll need to convert these files to UTF-8 or ASCII. One way we can convert the encoding is using Linux's `inconv` package, which is pre-installed in most distros.
#
# We can check the file's current encoding like so, where `FILE_NAME.txt` is the name of the file:
# ```bash
# file -i FILE_NAME.txt
# ```
#
# ```bash
# iconv -f ISO-8859-1 -t UTF-8//TRANSLIT FILE_NAME.txt -o FILE_NAME.txt
# ```

spam_file_paths = glob.iglob('../data/enron/**/spam/**')
df_spam = pd.DataFrame(columns=['text', 'label'])

for file_path in spam_file_paths:
    text = ''
    with open(file_path, 'r') as file:
        for line in file:
            text += line
        df_spam = pd.concat([df_spam, pd.DataFrame({'text': text, 'label': [1]})], ignore_index=True)

# ### Read Ham Emails:

ham_file_paths = glob.iglob('../data/enron/ham/**')
df_ham = pd.DataFrame(columns=['text', 'label'])

for file_path in ham_file_paths:
    text = ''
    with open(file_path, 'r') as file:
        for line in file:
            text += line

        df_ham = pd.concat([df_ham, pd.DataFrame({'text': text, 'label': [0]})], ignore_index=True)

# ### Explore Dataset:
#
# ##### Labels:
# **1**: spam
# **0**: ham

# Spam examples:
df_spam.head()

# Ham examples:
df_ham.head()

# +
# Set size of plot:
plt.rcParams["figure.figsize"] = (8, 8)

fig, ax = plt.subplots()
ax.bar(
    ['Ham', 'Spam'],
    [len(df_ham), len(df_spam)],
    color='blue',
    alpha=0.45
)

plt.title('Document Count')
plt.xlabel('Category')
plt.ylabel('Number of examples')
plt.show()
# -

len_ham = len(df_ham)
len_spam = len(df_spam)
print('Ham Count:', len_ham)
print('Spam Count:', len_spam)
print('Total:', len_spam + len_ham)

combined_dataset = pd.concat([df_ham, df_spam])
x_dataset = combined_dataset['text']
y_dataset = combined_dataset['label'].astype(int)

# +
# combined_dataset.to_csv('../data/enron1.csv', index=False, escapechar='\\')
# -

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=.2, random_state=12)

print('Size of train set:', len(x_train))
print('Size of test set:', len(x_test))

# ## Process Data:

# We only need lemmatizer from spaCy:
# see <https://spacy.io/usage/processing-pipelines#disabling>
nlp = spacy.load('en_core_web_trf',
                 disable=['tagger', 'parser', 'ner', 'entity_linker', 'entity_ruler', 'textcat', 'textcat_multilabel',
                          'attribute_ruler', 'senter', 'sentencizer', 'tok2vec', 'transformer']
                 )

# remove stop words (parameter scaling):
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# ## Support Vector Machine:

# +
model_SVM = Pipeline(
    [
        ('vectorizer', TfidfVectorizer(stop_words=spacy_stopwords)),
        ('classifier', LinearSVC())
    ])

model_SVM.fit(x_train, y_train)
# -

print('Train Accuracy:', model_SVM.score(x_train, y_train))
print('Test accuracy:', model_SVM.score(x_test, y_test))


def predict_files(spam_glob_path: str, ham_glob_path: str, model):
    spam_file_paths = glob.iglob(spam_glob_path)
    ham_file_paths = glob.iglob(ham_glob_path)

    _labels = []
    _predictions = []

    i = 0
    for file_paths in (ham_file_paths, spam_file_paths):
        for file_path in file_paths:
            _labels.append(i)
            with open(file_path, 'r') as file:
                predict = model.predict(file)
                if sum(predict) / len(predict) >= .5:
                    _predictions.append(1)
                else:
                    _predictions.append(0)
        i += 1
    return _predictions, _labels


predictions, labels = predict_files('../data/enron1/spam/**', '../data/enron1/ham/**', model_SVM)

f1_score(labels, predictions, average="macro")

# ### XGBoost

# +
model_xgb = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=spacy_stopwords)),
    ('classifier', XGBClassifier(objective='binary:logistic'))
])

model_xgb.fit(x_train, y_train)
# -

print('Train accuracy:', model_xgb.score(x_train, y_train))
print('Test accuracy:', model_xgb.score(x_test, y_test))

file_paths = glob.iglob('../data/enron1/spam/**')

predictions, labels = predict_files('../data/enron1/spam/**', '../data/enron1/ham/**', model_xgb)

f1_score(labels, predictions, average="macro")


