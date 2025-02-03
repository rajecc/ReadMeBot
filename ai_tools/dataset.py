import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Загрузить датасет книг из CSV файла.
    """
    dataset = pd.read_csv(file_path)
    return dataset

dataset = load_dataset("ai_tools/books_db.csv")