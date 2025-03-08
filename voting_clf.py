import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle

df = pd.read_csv("Docs/processed_data.csv")

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
training_set.to_csv("Docs/upsampled_set.csv", index=False)

# separating features and target again
X_train = training_set.drop(columns=['label'])
y_train = training_set.loc[:,'label']

#initialsizing individual models for the voting classifier
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
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

#analyze the model's performance
print(classification_report(y_test, y_pred))