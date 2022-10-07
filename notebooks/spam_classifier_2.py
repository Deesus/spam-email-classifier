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

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import spacy
from xgboost import XGBClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
import tensorflow as tf
# -

# ### Read spam emails:

spam_file_paths = glob.iglob('../input/harvardspamemailenron/data/spam/**')
df_spam = pd.DataFrame(columns=['text', 'label'])

for file_path in spam_file_paths:
    text = ''
    with open(file_path, 'r') as file:
        for line in file:
            text += line
        df_spam = pd.concat([df_spam, pd.DataFrame({'text': text, 'label': [1]})], ignore_index=True)

df_spam.head()

# ### Read ham emails:

ham_file_paths = glob.iglob('../input/harvardspamemailenron/data/ham/**')
df_ham = pd.DataFrame(columns=['text', 'label'])

for file_path in ham_file_paths:
    text = ''
    with open(file_path, 'r') as file:
        for line in file:
            text += line

        df_ham = pd.concat([df_ham, pd.DataFrame({'text': text, 'label': [0]})], ignore_index=True)

df_ham.head()

# + jupyter={"outputs_hidden": false}
# test_set = pd.read_csv('../data/enron_test_set.csv')
# train_set = pd.read_csv('../data/enron_train_set.csv')

# +
combined_dataset = pd.concat([df_ham, df_spam])

# Shuffle data:
combined_dataset = combined_dataset.sample(frac=1)

# +
train_test_split = int(len(combined_dataset) * 0.9)

train_set = combined_dataset[:train_test_split]
test_set = combined_dataset[train_test_split:]
# -

# ## Explore Data

# + jupyter={"outputs_hidden": true}
# Set size of plot:
plt.rcParams["figure.figsize"] = (8, 8)

fig, ax = plt.subplots()
ax.bar(
    ['Ham', 'Spam'],
    [len(df_ham), len(df_spam)],
    color='blue',
    alpha=0.72
)

plt.title('Document Count')
plt.xlabel('Category')
plt.ylabel('Number of examples')
plt.show()
# -

# Save data as csv:
# combined_dataset.to_csv('../working/harvard_enron_email.csv', index=False)
train_set.to_csv('../working/train_set.csv', index=False)
test_set.to_csv('../working/test_set.csv', index=False)

len_ham = len(df_ham)
len_spam = len(df_spam)
print('Ham Count:', len_ham)
print('Spam Count:', len_spam)
print('Total:', len_spam + len_ham)

# +
map_len = combined_dataset['text'].map(len)
max_length = map_len.max()
avg_length = map_len.sum() // len(combined_dataset)
median_length = int(map_len.median())
std_length = int(map_len.std())

print('Total number of examples:', len(combined_dataset))
print('Max length of document:', max_length)
print('Avg length of documents:', avg_length)
print('Median length of documents:', median_length)
print('Std of document length:', std_length)
# -

# ## Process Data:

# We only need lemmatizer from spaCy:
# see <https://spacy.io/usage/processing-pipelines#disabling>
nlp = spacy.load('en_core_web_trf',
                 disable=['tagger', 'parser', 'ner', 'entity_linker', 'entity_ruler', 'textcat', 'textcat_multilabel',
                          'attribute_ruler', 'senter', 'sentencizer', 'tok2vec', 'transformer']
                 )

# remove stop words (parameter scaling):
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# +
x_train = train_set['text']
y_train = train_set['label']

x_test = test_set['text']
y_test = test_set['label']
# -

# ## Support Vector Machine:

# +
model_SVM = Pipeline(
    [
        ('vectorizer', TfidfVectorizer(stop_words=spacy_stopwords)),
        ('classifier', LinearSVC())
    ])

model_SVM.fit(x_train, y_train)

# +
# svm_test_predictions = 
# svm_train_predictions = 
# -

print('Train Accuracy:', model_SVM.score(x_train, y_train))
print('Test accuracy:', model_SVM.score(x_test, y_test))

# ## Process Data for Transformer

dataset = Dataset.from_pandas(train_set)

# +
MAX_CHAR_LENGTH = 2000

percent_docs_over_2000 = combined_dataset['text'].map(lambda x: len(x) > MAX_CHAR_LENGTH).sum() / len(combined_dataset)
print(f'Percent of documents that have character length greater than 2,000: {percent_docs_over_2000 * 100:.2f}%')
# -

MODEL_NAME = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.shuffle(seed=2)


# +
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_CHAR_LENGTH)

tokenized_data = dataset.map(preprocess, batched=True, batch_size=8)
tokenized_data
# -

batch_size = 2
num_epochs = 15

batches_per_epoch = len(tokenized_data['train']) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-5, end_learning_rate=5e-9, decay_steps=total_train_steps
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/kaggle/working/checkpoints',
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True
)


# +
class ThresholdStoppageCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdStoppageCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["sparse_categorical_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True

# If validation accuracy reaches 99.7% accuracy, stop training:
threshold_stoppage_callback = ThresholdStoppageCallback(threshold=0.997)

# +
tf_train_set = tokenized_data['train'].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')
)

tf_validation_set = tokenized_data['test'].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')
)
# -

model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Free up some memory:
del tokenized_data
del dataset
del combined_dataset

# +
# TODO: update file path to local:
# model.load_weights('./data/saved_weights/distilroberta-base/weights_v2.h5')
# -

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model_history = model.fit(
    tf_train_set,
    validation_data=tf_validation_set,
    epochs=num_epochs,
    callbacks=[model_checkpoint_callback, threshold_stoppage_callback]
)

# Save weights:
model.save_weights('../working/saved_weights/distilroberta-base/weights_v1.h5')

# ## Model Evaluation:

# +
# Increase figure size:
plt.rcParams["figure.figsize"] = (12,8)

# Plot accuracy:
plt.plot(model_history.history['sparse_categorical_accuracy'], label='accuracy')
plt.plot(model_history.history['val_sparse_categorical_accuracy'], label='val_accuracy')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.show()


# -

def predict(text: str):
    encoded_input = tokenizer.encode(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_CHAR_LENGTH,
        return_tensors="tf"
    )
    
    # Make prediction from logits:
    prediction = model.predict(encoded_input)['logits']
    prediction = tf.nn.softmax(prediction)
    prediction = np.argmax(prediction)
    
    return prediction


test_predictions = [predict(x) for x in test_set['text']]
test_set_y = test_set['label'].to_list()

print('Testset F1 Score:', f1_score(test_set_y, test_predictions, average="macro"))
print('Testset Accuracy:', accuracy_score(test_set_y, test_predictions))
print('\n')
print(classification_report(test_set_y, test_predictions))

train_predictions = [predict(x) for x in train_set['text']]
train_set_y = train_set['label'].to_list()

print('Trainset F1 Score:', f1_score(train_set_y, train_predictions, average="macro"))
print('Trainset Accuracy:', accuracy_score(train_set_y, train_predictions))
print('\n')
print(classification_report(train_set_y, train_predictions))


