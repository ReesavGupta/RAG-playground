import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['transcript', 'label'])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    df['chunks'] = df['transcript'].apply(lambda x: splitter.split_text(x))
    return df