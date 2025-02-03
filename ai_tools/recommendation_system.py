from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from database.database import load_recommendation_history, save_recommendation_history

# Импорт функций анализа из analyze_system и сжатия из summarize_system
from ai_tools.analyze_system import extract_tags_and_genres
from ai_tools.summarize_system import compress_text

# Инициализация моделей перевода и сравнения
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Для вычисления косинусного сходства


# ------------------------------
# Функция для перевода текста с русского на английский
def translate_to_english(text: str) -> str:
    """
    Перевести текст с русского на английский.
    """
    input_text = f">>en<< {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = translation_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Translated to English: {translated_text}")
    return translated_text


# ------------------------------
# Функция для вычисления косинусного сходства между двумя текстами
def compute_similarity(text1: str, text2: str) -> float:
    """
    Вычислить косинусное сходство между двумя текстами с использованием эмбеддингов.
    """
    embeddings1 = similarity_model.encode([text1], convert_to_tensor=True)
    embeddings2 = similarity_model.encode([text2], convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_similarity.item()


# ------------------------------
# Функция для извлечения предпочтений из пользовательского ввода
def get_preferences_from_input(user_input: str) -> Dict[str, List[str]]:
    """
    Получить теги и жанры из пользовательского ввода.
    Перевод осуществляем, если требуется, затем анализируем текст.
    """
    # Переводим ввод (если база на английском, можно оставить теги на английском)
    translated_input = translate_to_english(user_input)
    extracted_data = extract_tags_and_genres(translated_input)
    return extracted_data


# ------------------------------
# Функция для извлечения предпочтений из выбранной книги
def get_preferences_from_history(selected_history: str) -> Dict[str, List[str]]:
    """
    Сжать описание выбранной книги из истории и извлечь теги и жанры.
    """
    # Сжимаем описание книги
    compressed_text = compress_text(selected_history)
    all_tags, all_genres = [], []

    # Извлекаем теги и жанры из сжатого текста
    extracted_data = extract_tags_and_genres(compressed_text)
    all_tags.extend(extracted_data.get('tags', []))
    all_genres.extend(extracted_data.get('genres', []))

    # Убираем дубликаты
    return {"tags": list(set(all_tags)), "genres": list(set(all_genres))}


# ------------------------------
# Функция для объединения предпочтений (если учтены и ввод, и история)
def combine_preferences(pref_input: Dict[str, List[str]], pref_history: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Объединить теги и жанры, полученные из пользовательского ввода и истории.
    """
    combined_tags = list(set(pref_input.get("tags", []) + pref_history.get("tags", [])))
    combined_genres = list(set(pref_input.get("genres", []) + pref_history.get("genres", [])))
    return {"tags": combined_tags, "genres": combined_genres}

def search_books_1_mode(selected_history, dataset):
    pref_history = get_preferences_from_history(selected_history)
    # Фильтрация по тегам и жанрам
    filtered_books = dataset[dataset['category'].apply(lambda x:
                                                       any(tag in x for tag in pref_history['tags']) or
                                                       any(genre in x for genre in pref_history['genres']))]
    # Сравнение с полными описаниями
    scored_books = []
    for _, row in filtered_books.iterrows():
        # Сравниваем описание книги с текстом пользователя и историей книг
        similarity_score_history = compute_similarity(selected_history, row['description'])
        if similarity_score_history > 0:
            scored_books.append(
                {'title': row['title'], 'similarity': similarity_score_history, 'description': row['description']})

    # Сортируем книги по убыванию схожести
    scored_books = sorted(scored_books, key=lambda x: x['similarity'], reverse=True)
    return scored_books[:5]

def search_books_2_mode(user_input, dataset):
    pref_input = get_preferences_from_input(user_input)
    # Фильтрация по тегам и жанрам
    filtered_books = dataset[dataset['category'].apply(lambda x:
                                                       any(tag in x for tag in pref_input['tags']) or
                                                       any(genre in x for genre in pref_input['genres']))]
    # Сравнение с полными описаниями
    scored_books = []
    for _, row in filtered_books.iterrows():
        # Сравниваем описание книги с текстом пользователя и историей книг
        similarity_score_input = compute_similarity(user_input, row['description'])
        if similarity_score_input > 0:
            scored_books.append(
                {'title': row['title'], 'similarity': similarity_score_input, 'description': row['description']})

    # Сортируем книги по убыванию схожести
    scored_books = sorted(scored_books, key=lambda x: x['similarity'], reverse=True)
    return scored_books[:5]

def search_books_3_mode(user_input, selected_history, dataset):
    """
    Поиск книг в базе с учетом категории и вычисления косинусного сходства между описаниями книг и текстами ввода/истории.
    """
    # Получаем предпочтения из ввода и истории
    pref_input = get_preferences_from_input(user_input)
    pref_history = get_preferences_from_history(selected_history)

    # Объединяем предпочтения
    combined_preferences = combine_preferences(pref_input, pref_history)

    # Формируем строку для поиска
    user_representation = user_input  # Текст ввода пользователя
    history_representation = selected_history  # Описание книги из истории

    # Отбираем книги по категориям (используем столбец 'category')
    filtered_books = dataset[dataset['category'].apply(lambda x:
                                                       any(tag in x for tag in combined_preferences['tags']) or
                                                       any(genre in x for genre in combined_preferences['genres']))]

    # Сравнение с полными описаниями
    scored_books = []
    for _, row in filtered_books.iterrows():
        # Сравниваем описание книги с текстом пользователя и историей книг
        similarity_score_input = compute_similarity(user_representation, row['description'])
        similarity_score_history = compute_similarity(history_representation, row['description'])

        # Общий балл схожести
        total_similarity = max(similarity_score_input, similarity_score_history)

        if total_similarity > 0:
            scored_books.append(
                {'title': row['title'], 'similarity': total_similarity, 'description': row['description']})

    # Сортируем книги по убыванию схожести
    scored_books = sorted(scored_books, key=lambda x: x['similarity'], reverse=True)
    return scored_books[:5]


# ------------------------------
# Функция для формирования рекомендаций по предпочтениям пользователя
def get_book_recommendations(user_input, selected_book, mode, user_id, history_exclude_option, dataset):
    """
    Формирует рекомендации по трём режимам:
      1. Учитывать и пользовательский ввод, и выбранные книги из истории.
      2. Учитывать только пользовательский ввод.
      3. Учитывать только выбранные книги из истории.
    Опционально исключает ранее рекомендованные книги.
    """

    recommendations = []
    if history_exclude_option:
        previously_recommended = load_recommendation_history(user_id)
        dataset = dataset[~dataset['title'].isin(previously_recommended)]

    if mode == 1:
        # Режим 1: объединение пользовательского ввода и истории
        recommendations = search_books_1_mode(selected_book, dataset)

    elif mode == 2:
        # Режим 2: учитывать только пользовательский ввод
        recommendations = search_books_2_mode(user_input, dataset)

    elif mode == 3:
        # Режим 3: учитывать только книги из истории
        recommendations = search_books_3_mode(user_input, selected_book, dataset)

    # Сохраняем историю рекомендованных книг
    save_recommendation_history(user_id, [book['title'] for book in recommendations])

    return recommendations

