import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample, shuffle
import subprocess

# Load dataset (assuming SecretBench format)
raw_df = pd.read_csv("Docs/secretbench.csv")
df_filtered = raw_df[raw_df["file_type"].isin(["java", "py", "js", "rb", "nix"])]
df_filtered.loc[:, 'secret'] = df_filtered.loc[:, 'secret'].str[1:-1]

SECRET_KEYWORDS = ["password", "token", "api", "key", "auth"]
SPECIAL_CHARACTERS = [':', '{', '}','def', 'class', 'do', 'return', 'end', '(', ')']

def format_js_code(js_code):
    formatted_code = subprocess.run(
        ["prettier", "--stdin-filepath", "file.js"],
        input=js_code.encode(),
        capture_output=True,
        text=True
    )
    return formatted_code.stdout if formatted_code.stdout else js_code

def extract_features(file_name, secret):

    with open("Docs/Files/" + file_name, "r", encoding="utf-8") as f:
        code = f.read()
        lines = f.readlines()
        
    if '.js' in file_name:
        format_js_code(code)
        lines = code.splitlines()
        
    context = ''
    secret_index = df_filtered.loc[df_filtered['secret'] == secret, 'start_line'] - 1
    for line in lines[secret_index::-1]:
        context = context.join(line)
        if any(char in line.lower() for char in SPECIAL_CHARACTERS):
            break
    context = reversed(context)
    for line in lines[secret_index:len(lines)-1]:
        context = context.join(line)
        if any(char in line.lower() for char in SPECIAL_CHARACTERS):
            break
    
    # Binary features for context
    context_features = {
        'has_secret_keyword': int(any(k in lines[secret_index].lower() for k in SECRET_KEYWORDS)),
        'has_comment': int('#' in lines[secret_index]),
        'has_assignment': int('=' in lines[secret_index]),
        'has_parenthesis': int('(' or ')' in secret),
        'has_brackets': int('[' or ']' in secret),
        'has_period': int('.' in secret),
        'has_space': int(' ' in secret),
    }

    return pd.DataFrame(context_features)

#df_filtered['context'] = df_filtered.apply(lambda row: extract_function_context(row['file_path'], row['start_line']), axis=1)
features = df_filtered.apply(lambda row: extract_features(row['file_identifier'], row['start_line']), axis=1)
df = pd.concat([features, df_filtered], axis=1)

for column in ['label', 'is_template', 'in_url', 'is_multiline']:
    df[column] = df[column].map({'Y':1, 'N':0})
    df[column] = pd.to_numeric(df[column])
    
#df['has_paranthesis'] = df['secret'].apply(lambda x: 1 if ('(' or ')') in x else 0)
#df['has_brackets'] = df['secret'].apply(lambda x: 1 if ('[' or ']') in x else 0)
#df['has_period'] = df['secret'].apply(lambda x: 1 if '.' in x else 0)
#df['has_keyword'] = df['secret'].apply(lambda x: 1 if word in x.lower() else 0 for word in SECRET_KEYWORDS)
#df['has_space'] = df['secret'].apply(lambda x: 1 if ' ' in x else 0)
df['in_config_path'] = df['file_path'].apply(lambda x: 1 if 'config' in x.lower() else 0)
df['is_testfile'] = int('test' or 'example' in df['file_path'].lower())
    
df.drop(
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
    inplace=True,
    axis=1
)

print(df)
df.info()

""" # Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Upsample only the training set
majority_class = train_df[train_df.label == 0]
minority_class = train_df[train_df.label == 1]

minority_upsampled = resample(minority_class, 
                               replace=True,    # Sample with replacement
                               n_samples=len(majority_class),  # Match majority class size
                               random_state=42)

train_df_balanced = pd.concat([majority_class, minority_upsampled])
train_df_balanced = shuffle(train_df_balanced, random_state=42)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Dataset class
class SecretDataset(Dataset):
    def __init__(self, dataframe):
        self.encodings = tokenizer(dataframe['context'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.features = torch.tensor(dataframe.drop(columns=['label', 'context']).values, dtype=torch.float)
        self.labels = torch.tensor(dataframe['label'].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['features'] = self.features[idx]
        item['labels'] = self.labels[idx]
        return item

train_dataset = SecretDataset(train_df_balanced)
test_dataset = SecretDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

print(classification_report(true_labels, preds)) """
