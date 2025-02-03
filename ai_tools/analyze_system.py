from typing import List, Dict
from huggingface_hub import InferenceClient

# Инициализация клиента для модели анализа (например, DeepSeek)
client = InferenceClient(provider="together", api_key="your_api_key")
model = "deepseek-ai/DeepSeek-V3"

# Глобальные переменные для хранения контекста книги и истории чата
book_context = ""
chat_history = []

# ------------------------------
# Функция для определения типа текста
def determine_text_type(text: str) -> str:
    """
    Определить тип текста (например, учебный, научно-популярный, научный, художественный)
    на основе его фрагмента. Возвращается один из типов.
    """
    # Берём средний фрагмент текста
    mid_point = len(text) // 2
    fragment = text[max(mid_point - 500, 0): mid_point + 500]
    messages = [
        {
            "role": "user",
            "content": (
                "Определи тип данного текста, выбери один из вариантов: учебный, научно-популярный, научный, художественный. "
                f"Вот фрагмент: {fragment}"
            )
        }
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100
    )
    text_type = completion.choices[0].message.content.strip()
    return text_type


# ------------------------------
# Функция для извлечения тегов и жанров из текста
def extract_tags_and_genres(text: str) -> Dict[str, List[str]]:
    """
    Извлечь теги и жанры из переданного текста.
    Теги извлекаются посредством запроса к модели, а жанры определяются эвристически.
    """
    # Извлечение тегов
    response_tags = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                    "Extract tags from the following text about books preferences. "
                    "Write only tags separated by commas in Russian: " + text
            )
        }],
        max_tokens=500
    )
    tags = [tag.strip() for tag in response_tags.choices[0].message.content.split(",") if tag.strip()]

    # Эвристическое определение жанров на основе ключевых слов
    possible_genres = {
        "учебный": ["образование", "учебный"],
        "научно-популярный": ["наука", "популярный"],
        "научный": ["исследование", "теория"],
        "художественный": ["роман", "рассказ"]
    }
    genres = []
    text_lower = text.lower()
    for genre, keywords in possible_genres.items():
        if any(keyword in text_lower for keyword in keywords):
            genres.append(genre)
    if not genres:
        genres.append("художественный")

    return {"tags": tags, "genres": genres}


# ------------------------------
# Функция для ответа на вопрос пользователя с регулируемой длиной ответа
def ask_question(question: str, answer_mode: str = "detailed") -> str:
    """
    Ответить на вопрос пользователя по содержанию книги.
    Параметр answer_mode задаёт режим длины ответа:
      - "short"   : краткий ответ,
      - "medium"  : средний ответ,
      - "detailed": развернутый ответ (по умолчанию).

    Если режим не "detailed", то полученный подробный ответ сжимается с использованием системы сжатия.
    """
    global chat_history
    chat_history.append({"role": "user", "content": question})

    messages = [
                   {"role": "system",
                    "content": (
                        "Ты — AI-ассистент, специализирующийся на анализе содержания книг. "
                        "Отвечай подробно на вопросы, анализируй стилистику, символизм и темы произведения. "
                        "Отвечай на русском языке."
                    )},
                   {"role": "user", "content": f"Вот текущий контекст книги:\n{book_context}"}
               ] + chat_history

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    detailed_answer = completion.choices[0].message.content

    # Если режим отличается от "detailed", сжимаем ответ до нужного объёма
    if answer_mode != "detailed":
        # Определяем желаемый итоговый размер (например, 50 слов для короткого, 100 для среднего)
        if answer_mode == "short":
            target_size = 50
        elif answer_mode == "medium":
            target_size = 100
        else:
            target_size = 200  # запасной вариант для неизвестного режима

        # Локальный импорт функции сжатия из summarize_system для избежания циклического импорта
        from ai_tools.summarize_system import compress_text
        adjusted_answer = compress_text(detailed_answer, final_target_size=target_size)
        final_answer = adjusted_answer
    else:
        final_answer = detailed_answer

    chat_history.append({"role": "assistant", "content": final_answer})
    return final_answer


def update_book_content(user_id: str, book_name: str, current_page: int, final_target_size: int = 100):
    """
    Дополняет контекст книги сжатыми последними прочитанными страницами.
    """
    global book_context
    
    # Получаем полный текст книги
    full_text = get_book_full_text(book_name, user_id).split()  # Разделяем на слова для имитации страниц
    total_pages = get_total_pages("books", book_name)  # Вычисляем общее количество страниц
    
    # Определяем последнюю обновленную страницу
    last_update_page = users_db[user_id]["reading_state"][book_name]["update_page"]
    
    # Если текущая страница последняя в книге, берем текст от последнего обновления до конца
    if current_page >= total_pages:
        recent_text = " ".join(full_text[(last_update_page - 1) * 500:])
    else:
        recent_text = " ".join(full_text[(last_update_page - 1) * 500: current_page * 500])
    
    if text_type == None:
        text_type = determine_text_type(recent_text)
    
    # Сжимаем текст
    compressed_text = compress_text(recent_text, final_target_size=final_target_size, text_type=text_type)
    
    # Дополняем контекст книги
    book_context += "\n" + compressed_text
    
    # Обновляем отметку последнего обновления
    users_db[user_id]["reading_state"][book_name]["update_page"] = current_page
    
    # Сохраняем обновленные данные
    save_users_db(users_db)
    
    return book_context

# ------------------------------
# Функция для сброса истории чата
def reset_chat():
    """
    Сбросить историю чата.
    """
    global chat_history
    chat_history = []

