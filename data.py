import pandas as pd

# Load dataset (assuming SecretBench format)
raw_db = pd.read_csv("Docs/secretbench.csv")
raw_db.loc[:, 'secret'] = raw_db.loc[:, 'secret'].str[1:-1]
raw_db = raw_db[raw_db["file_type"].isin(["java", "py", "js", "rb", "nix"])]

SECRET_KEYWORDS = ["password", "token", "api", "key", "auth"]
SPECIAL_CHARACTERS = [':', '{', '}','def', 'class', 'do', 'return', 'end', '(', ')']

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

df['has_paranthesis'] = raw_db['secret'].apply(lambda x: 1 if ('(' or ')') in x else 0)
df['has_brackets'] = raw_db['secret'].apply(lambda x: 1 if ('[' or ']') in x else 0)
df['has_period'] = raw_db['secret'].apply(lambda x: 1 if '.' in x else 0)
df['has_password'] = raw_db['secret'].apply(lambda x: 1 if 'Password' in x else 0)
df['has_space'] = raw_db['secret'].apply(lambda x: 1 if ' ' in x else 0)
df['in_config_path'] = raw_db['file_path'].apply(lambda x: 1 if 'config' in x.lower() else 0)
df['in_test_path'] = raw_db['file_path'].apply(lambda x: 1 if ('test' or 'example') in x.lower() else 0)

#converting datatypes of columns to (binary) numerical
for column in ['label', 'is_template', 'in_url', 'is_multiline']:
    df[column] = df[column].map({'Y':1, 'N':0})
    df[column] = pd.to_numeric(df[column])
    
df.to_csv("Docs/processed_data.csv", index=False)
