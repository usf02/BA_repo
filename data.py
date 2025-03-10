import pandas as pd
import re
import subprocess

# Load dataset (assuming SecretBench format)
raw_db = pd.read_csv("Docs/secretbench.csv")
raw_db.loc[:, 'secret'] = raw_db.loc[:, 'secret'].str[1:-1]
raw_db = raw_db[raw_db["file_type"].isin(["java", "py", "rb", "nix", "js"])]

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

print(df.info())
#df.to_csv("Docs/processed_data.csv", index=False)

#the function extracts the context of the secret (3 lines before and after) from the source file
def extract_context(id):
    file_name = raw_db.loc[raw_db['id'] == id, 'file_identifier'].iloc[0]
    secret_start = raw_db.loc[raw_db['id'] == id, 'start_line'].iloc[0] - 1
    secret_end = raw_db.loc[raw_db['id'] == id, 'end_line'].iloc[0] - 1
    secret = raw_db.loc[raw_db['id'] == id, 'secret'].iloc[0]
    
    try:
        with open("Docs/Files/" + file_name,"r", encoding='utf8') as f:
            lines = f.readlines()
            f.close()
                    
        if ('.js' in file_name):
            # Check if the file is minified
            avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
            is_minified = avg_line_length > 200  # Adjust threshold if needed 

            if is_minified:
                if len(lines) > 1:
                    content = ''.join(lines)
                else:
                    content = lines[0]
                
                # Attempt to split intelligently if the file is minified
                lines = re.split(r'(;|\{|,)', content)

                # Merge back to keep structure
                structured_lines = []
                temp_line = ""
                for segment in lines:
                    temp_line += segment
                    if segment in {';', '{', ','}:
                        structured_lines.append(temp_line + '\n')
                        temp_line = ""
                if temp_line:
                    structured_lines.append(temp_line + '\n')
                
                lines = structured_lines
                for i, line in enumerate(lines):
                    if (secret == line) or (secret in line):
                        secret_start = i
                        secret_end = i

            # Extract the desired context
            start_index = max(secret_start - 3, 0)
            end_index = min(secret_end + 4, len(lines))
            context = ''.join(lines[start_index:end_index])

            return context
        else:
            start_index = max(secret_start - 3, 0)
            end_index = min(secret_end + 4, len(lines))
            context = ''.join(lines[start_index:end_index])
            return context
    
    except FileNotFoundError:
        return f"File {file_name} not found"
    except Exception as e:
        return f"Error: {e}"

#apply the function to the dataset
df['context'] = raw_db.apply(lambda row: extract_context(row['id']), axis=1)
df.drop(df[df["context"].str.contains(".java not found", na=False)].index, inplace=True)

print(df.info())
#df.to_csv("Docs/processed_data_context.csv", index=False)
