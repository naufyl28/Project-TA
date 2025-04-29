import pandas as pd
import re

def clean_comment(text):
    text = str(text).lower()  
    text = re.sub(r'<.*?>', '', text)  # Menghapus tag HTML
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def filter_meaningless_comments(df, text_column):
    """
    Memfilter komentar yang memiliki panjang kurang dari 3 karakter.
    """
    df = df[df[text_column].str.len() > 2]
    return df

def load_and_preprocess_data(file_path='data/Data TA_LABEL.csv'):
    df = pd.read_csv(file_path)
    df['cleaned_comment'] = df['textDisplay'].apply(clean_comment)

    # Memfilter komentar yang kosong atau terlalu pendek
    df = filter_meaningless_comments(df, 'cleaned_comment')

    return df[['cleaned_comment', 'label']]


df = pd.read_csv('data/Data TA_LABEL.csv')
df['cleaned_comment'] = df['textDisplay'].apply(clean_comment)

df = filter_meaningless_comments(df, 'cleaned_comment')

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label_manual'].map(label_mapping)

df_cleaned = df[['cleaned_comment', 'label_manual']]

df_cleaned.to_csv('data/Data_TA_LABEL_cleaned.csv', index=False)

print("Data cleaned successfully.")