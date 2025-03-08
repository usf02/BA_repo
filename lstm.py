import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Input, Concatenate, Dropout, Lambda

df = pd.read_csv("Docs/processed_data_context.csv")

#split the dataset into training and test sets
X = df.drop(labels=['label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#joining the training and test sets for easier manipulation, then defining the majority and minority classes
training_set = pd.concat([X_train, y_train], axis=1)
majority_class = training_set.loc[training_set['label'] == 0]
minority_class = training_set.loc[training_set['label'] == 1]

#upsampling the minority class in the training set to match the majority class
minority_upsampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=42
)

#joining both classes back together and shuffling
training_set = pd.concat([majority_class, minority_upsampled], axis=0)
training_set = shuffle(training_set, random_state=42)
training_set.to_csv("Docs/upsampled_training_set.csv", index=False)

# separating features and target again
X_train = training_set.drop(columns=['label'])
y_train = training_set.loc[:,'label']

#encoding secrets and context using a tokenizer and adding padding to ensure uniform length
tokenizer = Tokenizer(num_words=657, oov_token="<OOV>")
tokenizer.fit_on_texts(training_set['context'])
encoded_train_contexts = tokenizer.texts_to_sequences(training_set['context'])
padded_train_contexts = pad_sequences(encoded_train_contexts, padding='post')
encoded_test_contexts = tokenizer.texts_to_sequences(X_test['context'])
padded_test_contexts = pad_sequences(encoded_test_contexts, padding='post')
X_test_num = X_test.drop(columns=['context']).to_numpy()
#max_index = max([max(seq) for seq in padded_contexts])

X_train_context = np.concatenate([X_train.to_numpy(), padded_train_contexts], axis=1)
X_test_context = np.concatenate([X_test_num, padded_test_contexts], axis=1)
y_train_context = y_train.to_numpy()
y_test_context = y_test.to_numpy()

model = Sequential()
model.add(Embedding(input_dim=9476, output_dim=128))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(192, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
model.fit(X_train_context, y_train_context, epochs=15, validation_split=0.2, class_weight={0: 1.0, 1: 3.0})

y_pred_prob = model.predict(X_test_context)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test_context, y_pred))