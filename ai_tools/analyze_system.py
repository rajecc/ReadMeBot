from googletrans import Translator
from typing import List, Dict
from huggingface_hub import InferenceClient
from database.database import get_book_full_text, save_users_db, get_total_pages

# Инициализация клиента для модели анализа (например, DeepSeek)
client = InferenceClient(provider="hf-inference", api_key="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")
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# ------------------------------
# Функция для определения типа текста
def determine_text_type(text: str) -> str:
    """
    Определить тип текста (например, учебный, научно-популярный, научный, художественный)
    на основе его фрагмента. Возвращается один из типов.
    """
    PROMPT_TEMPLATE = '''Analyze the given text and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.  

        Output the result strictly as a comma-separated list of parameters in russian without any additional text.  

        Examples of key parameters for different text types:  
        - **Literary: персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды  
        - **Scientific: основные термины, гипотезы, методы, доказательства, выводы  
        - **Journalistic: ключевые события, участники, место, время, аргументы  
        - **Technical: предмет описания, термины, инструкции, алгоритмы, примеры  
        - **Philosophical: основные идеи, аргументы, философские термины, парадоксы, концепции  

        Text:  
        "{text}"  
        
        '''
    # Берём средний фрагмент текста
    mid_point = len(text) // 2
    fragment = text[max(mid_point - 500, 0): mid_point + 500]

    messages = [
        {
            "role": "user",
            "content": (
                PROMPT_TEMPLATE.format(text=fragment)
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
                    "Write only tags separated only by commas.One word for one tag. The first letter of each tag must be capitalized.: " + text
            )
        }],
        max_tokens=100
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
def ask_question(question: str, book_context: str, chat_history: list ,answer_mode: str = "detailed") -> str:
    """
    Ответить на вопрос пользователя по содержанию книги.
    Параметр answer_mode задаёт режим длины ответа:
      - "short"   : краткий ответ,
      - "medium"  : средний ответ,
      - "detailed": развернутый ответ (по умолчанию).

    Если режим не "detailed", то полученный подробный ответ сжимается с использованием системы сжатия.
    """
    messages = [
                   {"role": "system",
                    "content":
                        "You are an AI assistant specializing in analyzing the content of books. Answer questions in detail, analyzing the style, symbolism, and themes of the work. Answer very shortly. Answer only in Russian."
                    },] + chat_history + [{ "role": "user","content": f"Here is the current context of the book:\n{book_context}\nHere is the question: {question}"}
]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300
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
    return final_answer


def update_book_content(user_id: str, book_name: str, current_page: int, last_update_page: int, book_context: str = "",final_target_size: int = 200):
    """
    Дополняет контекст книги сжатыми последними прочитанными страницами.
    """
    
    # Получаем полный текст книги
    full_text = get_book_full_text(book_name, user_id).split()  # Разделяем на слова для имитации страниц
    total_pages = get_total_pages("books", book_name)  # Вычисляем общее количество страниц
    
    # Если текущая страница последняя в книге, берем текст от последнего обновления до конца
    if current_page >= total_pages:
        recent_text = " ".join(full_text[(last_update_page - 1) * 500:])
    else:
        recent_text = " ".join(full_text[(last_update_page - 1) * 500: current_page * 500])
    
    text_type = determine_text_type(recent_text)
    
    # Сжимаем текст
    from ai_tools.summarize_system import compress_text
    compressed_text = compress_text(recent_text, final_target_size=final_target_size, text_type=text_type)

    print(compressed_text)
    # Дополняем контекст книги
    book_context += "\n" + compressed_text
    
    return book_context



