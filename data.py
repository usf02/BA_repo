import pandas as pd
import re

# Load dataset (assuming SecretBench format)
raw_db = pd.read_csv("Docs/secretbench.csv")
raw_db.loc[:, 'secret'] = raw_db.loc[:, 'secret'].str[1:-1]
raw_db = raw_db[raw_db["file_type"].isin(["js", "java", "py", "rb", "nix"])]

NULL_KEYWORDS = ['null', 'nil', 'undefined', 'none', 'true', 'false']

#create a new df with select features
df = raw_db.drop(
    labels=[
        #'id',
        #'secret',
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
df['has_tag'] = raw_db['secret'].apply(lambda x: 1 if 'begin private key' in x.lower() else 0 )
df['startswith_$'] = raw_db['secret'].apply(lambda x: 1 if x.startswith('$') else 0 )
df['has_null'] = raw_db['secret'].apply(lambda x: 1 if any(word in x.lower() for word in NULL_KEYWORDS) else 0)
df['has_arrow'] = df['secret'].apply(lambda x: 1 if '->' in x else 0)
df['is_numerical'] = df['secret'].apply(lambda x: 1 if x.isdigit() else 0)
df['in_config_path'] = raw_db['file_path'].apply(lambda x: 1 if 'config' in x.lower() else 0)
df['in_test_path'] = raw_db['file_path'].apply(lambda x: 1 if ('test' or 'example') in x.lower() else 0)

#converting datatypes of columns to (binary) numerical
for column in ['label', 'is_template', 'in_url', 'is_multiline']:
    df[column] = df[column].map({'Y':1, 'N':0})
    df[column] = pd.to_numeric(df[column])

#df.to_csv("Docs/raw_set.csv", index=False)

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
            is_minified = avg_line_length > 200

            if is_minified:
                if len(lines) > 1:
                    content = ''.join(lines)
                else:
                    content = lines[0]
                
                # Attempt to split manually if the file is minified
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
                    if secret in line:
                        secret_start = i
                        secret_end = i

            # Extract the desired context
            start_index = max(secret_start - 3, 0)
            end_index = min(secret_end + 4, len(lines))
            full_context = ''.join(lines[start_index:end_index])
            
            lines_before = ''.join(lines[start_index:secret_start])
            lines_after = ''.join(lines[secret_end+1:end_index])
            context = ''.join([lines_before, lines_after])

            return full_context, context, lines_before, lines_after
        else:
            start_index = max(secret_start - 3, 0)
            end_index = min(secret_end + 4, len(lines))
            full_context = ''.join(lines[start_index:end_index])

            lines_before = ''.join(lines[start_index:secret_start])
            lines_after = ''.join(lines[secret_end+1:end_index])
            context = ''.join([lines_before, lines_after])
            
            return full_context, context, lines_before, lines_after
    
    except FileNotFoundError:
        return f"File {file_name} not found"
    except Exception as e:
        return f"Error: {e}"

#apply the function to the dataset
df[['full_context', 'context_w/o_secret', 'context_before', 'context_after']] = raw_db.apply(lambda row: pd.Series(extract_context(row['id'])), axis=1)
df.info()

df.drop(df[df['full_context'].str.contains(".java not found", na=False)].index, inplace=True)
print(df.duplicated(subset=['secret', 'full_context']).sum())
df.drop_duplicates(subset=['secret', 'full_context'], keep='first', inplace=True)
df.dropna(inplace=True, ignore_index=True)
df.info()

df.to_json("Docs/dataset.json", orient='records', lines=False, indent=4)
