import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

pd.options.mode.copy_on_write = True
pd.set_option('future.no_silent_downcasting', True)

#load and process the dataset
raw_db = pd.read_csv("Docs/secretbench.csv")
raw_db.loc[:, 'secret'] = raw_db.loc[:, 'secret'].str[1:-1]
raw_db = raw_db[raw_db["file_type"].isin(["java", "py", "js", "rb"])]

#create a new df with select features
df = raw_db.drop(
    labels=[
        'id',
        'secret',
        'repo_name',
        'domain',
        'commit_id',
        'file_path',
        'file_type',
        'start_line',
        'end_line',
        'start_column',
        'end_column',
        'committer_email',
        'commit_date',
        'character_set',
        'has_words',
        'category',
        'file_identifier',
        'repo_identifier',
        'comment'],
    inplace=False,
    axis=1
)

file_types = ['java', 'py', 'js', 'rb']
for type in file_types:
    df['is_'+type] = raw_db['file_type'].apply(lambda x: 1 if x == type else 0)
 
df['has_paranthesis'] = raw_db['secret'].apply(lambda x: 1 if ('(' or ')') in x else 0)
df['has_brackets'] = raw_db['secret'].apply(lambda x: 1 if ('[' or ']') in x else 0)
df['has_period'] = raw_db['secret'].apply(lambda x: 1 if '.' in x else 0)
df['has_Password'] = raw_db['secret'].apply(lambda x: 1 if 'Password' in x else 0)
df['has_space'] = raw_db['secret'].apply(lambda x: 1 if ' ' in x else 0)

#converting datatypes of columns to (binary) numerical
for column in ['label', 'is_template', 'in_url', 'is_multiline']:
    df[column] = df[column].map({'Y':1, 'N':0})
    df[column] = pd.to_numeric(df[column])

#defining the function that extracts the context of the secret (3 lines before and after) from the source file
def extract_context(file_name, secret):
    try:
        with open("Docs/Files/" + file_name,"r", encoding='utf8') as f:
            lines = f.readlines()
            f.close()
            
        context = ''
        if (len(lines) == 1):
            context = lines[0]
            secret_index = context.find(secret)
            start_index = max(secret_index - 100, 0)
            end_index = min(secret_index + len(secret) + 100, len(context))
            context = context[start_index:end_index]
            return context
        else:
            secret_index = raw_db.loc[raw_db['secret'] == secret, 'start_line'] - 1
            start_index = max(secret_index + 3, 0)
            end_index = min(secret_index + 4, len(lines))
            context = context.join(lines[start_index:end_index])
            return context
    
    except FileNotFoundError:
        return f"File {file_name} not found"
    except Exception as e:
        return f"Error: {e}"

#apply the function to the dataset
df['context'] = raw_db.apply(lambda row: extract_context(row['file_identifier'], row['secret']), axis=1)

#split the dataset into training and test sets
X = df.drop(labels=['label'], axis=1)
y = df.loc[:,'label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#joining the training and test sets for easier manipulation, then defining the majority and minority classes
training_set = pd.concat([X_train, y_train], axis=1)
#training_set.info()
#print(training_set.loc[:, 'label'].value_counts())
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
balanced_training_set = pd.concat([majority_class, minority_upsampled], axis=0)
balanced_training_set = shuffle(balanced_training_set, random_state=42)
#balanced_training_set.info()

# separating features and target again
balanced_X_train = balanced_training_set.drop(columns=['label', 'context'])
balanced_y_train = balanced_training_set.loc[:,'label']

#encoding secrets and context using a tokenizer and adding padding to ensure uniform length
tokenizer = Tokenizer(num_words=657, oov_token="<OOV>")
tokenizer.fit_on_texts(balanced_training_set['context'])
encoded_train_contexts = tokenizer.texts_to_sequences(balanced_training_set['context'])
padded_train_contexts = pad_sequences(encoded_train_contexts, padding='post')
encoded_test_contexts = tokenizer.texts_to_sequences(X_test['context'])
padded_test_contexts = pad_sequences(encoded_test_contexts, padding='post')
X_test.drop(columns=['context'], inplace=True)
#max_index = max([max(seq) for seq in padded_contexts])
#balanced_X_train.info()

""" #initialsizing individual models for the voting classifier
lr = LogisticRegression(max_iter=1000, random_state=42)
nb = GaussianNB()
svm = SVC(probability=True, random_state=42)

#combining models
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('nb', nb),
        ('svm', svm)],
    voting='soft',
    weights=[1,1,1]
)

#training the model then making predictions
voting_clf.fit(balanced_X_train, balanced_y_train)
y_pred = voting_clf.predict(X_test)

#analyze the model's performance
print(classification_report(y_test, y_pred)) """

X_train_context = np.concatenate([balanced_X_train.to_numpy(), padded_train_contexts], axis=1)
X_test_context = np.concatenate([X_test.to_numpy(), padded_test_contexts], axis=1)
y_train_context = balanced_y_train.to_numpy()
y_test_context = y_test.to_numpy()

model = Sequential()
model.add(Embedding(input_dim=9476, output_dim=128))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(192, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall',])
model.fit(X_train_context, y_train_context, epochs=15, validation_split=0.2)

y_pred_prob = model.predict(X_test_context)
y_pred = (y_pred_prob > 0.3).astype(int)

print(classification_report(y_test_context, y_pred))