import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

df = pd.read_json("Docs/raw_set_contextRemovedDuplicatesUnordered.json")
df.dropna(inplace=True, ignore_index=True)

#split the dataset into training and test sets
X = df.drop(columns=['label', 'secret', 'commit_date'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

training_set = pd.concat([X_train, X_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)
for df in [X_train, X_test]:
    df.drop(columns='id', inplace=True)

# separating features and target again and preparing features for model
X_train_num = X_train.drop(columns=['context'])
X_test_num = X_test.drop(columns=['context'])
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#encoding secrets and context using a tokenizer and adding padding to ensure uniform length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['context'])
vocab_size = len(tokenizer.word_index) + 1
max_token_length = 200

train_context_enc = tokenizer.texts_to_sequences(X_train['context'])
train_context_pad = pad_sequences(train_context_enc, padding='post', maxlen=max_token_length)

test_context_enc = tokenizer.texts_to_sequences(X_test['context'])
test_context_pad = pad_sequences(test_context_enc, padding='post', maxlen=max_token_length)

X_train = np.concatenate([X_train_num.to_numpy(), train_context_pad], axis=1)
X_test = np.concatenate([X_test_num.to_numpy(), test_context_pad], axis=1)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(256, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
test_set['prediction'] = y_pred
test_set.to_json("Docs/lstm-10epochs/pred.json", orient='records', lines=False, indent=4)

report = classification_report(y_test, y_pred)
with open("Docs/lstm-10epochs/results.txt", 'w') as f:
    f.write(report)
    f.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['false secret', 'true secret'], yticklabels=['false secret', 'true secret'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Dataset')
plt.tight_layout()
plt.savefig('Docs/lstm-10epochs/cm.png')
plt.show()