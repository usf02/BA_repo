import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("Docs/raw_set.csv")

#split the dataset into training and test sets
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

training_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)
for df in [X_train, X_test]:
    df.drop(columns='id', inplace=True)

#initialsizing individual models for the voting classifier
lr = LogisticRegression(max_iter=1000)
nb = GaussianNB()
svm = SVC(probability=True)

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
joblib.dump(voting_clf, 'voting_clf.pk1')
y_pred = voting_clf.predict(X_test)
test_set['predictions'] = y_pred
test_set.to_csv("Docs/clf_predictions.csv", index=False)

#analyze the model's performance
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['false secret', 'true secret'], yticklabels=['false secret', 'true secret'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Dataset')
plt.tight_layout()
plt.show()