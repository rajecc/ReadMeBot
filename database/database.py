import os
import json
from PyPDF2 import PdfReader
import chardet

# Путь к файлу базы данных пользователей
USERS_DB_PATH = "users_db.json"

# ------------------------------
# Функции работы с базой данных пользователей

def load_users_db():
    """
    Загружает базу данных пользователей из файла.
    Если файла нет, возвращает пустой словарь.
    """
    if not os.path.exists(USERS_DB_PATH):
        return {}
    with open(USERS_DB_PATH, "r", encoding="utf-8") as db_file:
        return json.load(db_file)

def save_users_db(users_db):
    """
    Сохраняет базу данных пользователей в файл.
    """
    with open(USERS_DB_PATH, "w", encoding="utf-8") as db_file:
        json.dump(users_db, db_file, ensure_ascii=False, indent=4)

def update_reading_state(user_id, book_name, page, total_pages):
    """
    Обновляет состояние чтения пользователя.
    """
    user_id = str(user_id)
    users_db = load_users_db()

    if user_id not in users_db:
        users_db[user_id] = {"books": {}, "reading_state": {}, "preferences": {}}

    users_db[user_id]["reading_state"][book_name] = {"page": page, "total_pages": total_pages}

    save_users_db(users_db)
    return users_db[user_id]["reading_state"][book_name]

# ------------------------------
# Функции работы с книгами

def get_total_pages(user_books_dir, book_name, page_size=1000):
    """
    Вычисляет общее количество страниц книги.
    """
    file_path = os.path.join(user_books_dir, book_name)
    try:
        with open(file_path, "rb") as book_file:
            book_file.seek(0, os.SEEK_END)
            total_size = book_file.tell()
            total_pages = total_size // page_size + (1 if total_size % page_size > 0 else 0)
            return total_pages
    except Exception:
        return 0

def get_book_page(user_books_dir: str, book_name: str, page: int, page_size: int = 1000) -> str:
    """
    Получает текст определенной страницы книги.
    """
    file_path = os.path.join(user_books_dir, book_name)
    try:
        with open(file_path, "r", encoding="utf-8") as book_file:
            book_file.seek(page * page_size)
            content = book_file.read(page_size)
            return content if content else "Вы достигли конца книги."
    except FileNotFoundError:
        return "Книга не найдена."
    except UnicodeDecodeError:
        with open(file_path, "rb") as book_file:
            raw_data = book_file.read()
            detected_encoding = chardet.detect(raw_data)["encoding"]
            content = raw_data.decode(detected_encoding)
            start = page * page_size
            return content[start:start + page_size]
    except Exception as e:
        return f"Ошибка чтения книги: {str(e)}"

def get_current_page(user_id, book_name):
    users_db = load_users_db()
    user_id = str(user_id)
    if user_id in users_db and book_name in users_db[user_id]["books"]:
        return users_db[user_id]["books"][book_name]["page"]
    return None  # Если книги нет, возвращаем None

def get_book_full_text(book_name: str, user_id: str) -> str:
    """
    Получает полный текст книги.
    """
    book_path = os.path.join("books", str(user_id), book_name)
    file_extension = book_name.split('.')[-1].lower()

    if file_extension == 'txt':
        try:
            with open(book_path, 'rb') as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']
                return raw_data.decode(detected_encoding)
        except Exception as e:
            return f"Ошибка при чтении текстового файла: {e}"

    elif file_extension == 'pdf':
        try:
            with open(book_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                return " ".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            return f"Ошибка при чтении PDF файла: {e}"

    return "Неподдерживаемый формат файла."

# ------------------------------
# Функции работы с предпочтениями пользователя

# Функция для добавления рекомендаций в историю пользователя
def save_recommendation_history(user_id, recommendations):
    users_db = load_users_db()
    user_id = str(user_id)
    users_db[user_id]['recommendation_history'].extend(recommendations)
    save_users_db(users_db)

# Функция для загрузки истории рекомендаций пользователя
def load_recommendation_history(user_id):
    users_db = load_users_db()
    user_id = str(user_id)
    history = users_db[user_id]['recommendation_history']
    return history