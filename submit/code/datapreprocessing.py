import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
df_cm = pd.read_csv('data/all.cm', delimiter='\t', header=None, names=['ID', 'Command'])
df_nl = pd.read_csv('data/all.nl', delimiter='\t', header=None, names=['ID', 'Description'])

# 检查数据加载情况
print(f"Number of rows in Commands dataset: {len(df_cm)}")
print(f"Number of rows in Descriptions dataset: {len(df_nl)}")

# 清理 ID 列，确保格式一致
df_cm['ID'] = df_cm['ID'].str.strip().str.lower()
df_nl['ID'] = df_nl['ID'].str.strip().str.lower()

# 检查唯一性和重复值
print("Unique IDs in Commands dataset:", len(df_cm['ID'].unique()))
print("Unique IDs in Descriptions dataset:", len(df_nl['ID'].unique()))
print("Duplicate IDs in Commands dataset:", df_cm['ID'].duplicated().sum())
print("Duplicate IDs in Descriptions dataset:", df_nl['ID'].duplicated().sum())

# 去重
df_cm = df_cm.drop_duplicates(subset='ID')
df_nl = df_nl.drop_duplicates(subset='ID')

# 分析未匹配的 ID
commands_ids = set(df_cm['ID'])
descriptions_ids = set(df_nl['ID'])

only_in_commands = commands_ids - descriptions_ids
only_in_descriptions = descriptions_ids - commands_ids

print(f"IDs in Commands but not in Descriptions: {len(only_in_commands)}")
print(f"IDs in Descriptions but not in Commands: {len(only_in_descriptions)}")

# 清理命令和描述
def advanced_normalize_bash_command(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP_ADDRESS>', text)
    text = re.sub(r'/[^\s]+', '<PATH>', text)
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'-[a-zA-Z]', '<FLAG>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_normalize_nl_description(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'\(.*?\)', '', text).strip()
    return text

df_cm['cleaned_command'] = df_cm['Command'].apply(advanced_normalize_bash_command)
df_nl['cleaned_description'] = df_nl['Description'].apply(advanced_normalize_nl_description)

# 合并数据
merged_df = pd.merge(df_cm, df_nl, on='ID', how='outer')
merged_df.fillna("<MISSING>", inplace=True)
print(f"Number of rows in merged DataFrame (outer join): {len(merged_df)}")

# 检查缺失情况
print("Rows with missing Commands:", len(merged_df[merged_df['cleaned_command'] == '<MISSING>']))
print("Rows with missing Descriptions:", len(merged_df[merged_df['cleaned_description'] == '<MISSING>']))

# 提取关键词
def extract_keywords(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return ' '.join(keywords)
    except ValueError:
        return ""

merged_df['description_keywords'] = merged_df['cleaned_description'].apply(extract_keywords)

# 保存清理后的数据
merged_df[['ID', 'cleaned_command']].to_csv('data/preprocessed_all.cm', sep='\t', header=False, index=False)
merged_df[['ID', 'cleaned_description']].to_csv('data/preprocessed_all.nl', sep='\t', header=False, index=False)

print("清理后的数据已保存到 'preprocessed_all.cm' 和 'preprocessed_all.nl' 文件中。")
