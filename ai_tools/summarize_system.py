import os
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

BOOKS_DIRECTORY = "books"

# Инициализируем клиента для модели сжатия (например, Qwen)
client = InferenceClient(
    provider="hf-inference",
    api_key="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"
)

# Инициализация модели для эмбеддингов (если понадобится)
split_model = SentenceTransformer('all-MiniLM-L6-v2')


# ------------------------------
# Функция для разделения текста на чанки по количеству слов
def split_into_chunks(sentences: List[str], max_size: int) -> List[str]:
    """
    Разбить список предложений на чанки, суммарное число слов в которых не превышает max_size.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_size += len(sentence.split())
        if current_size >= max_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


# ------------------------------
# Функция для сжатия одной части текста до заданного размера с помощью модели через API
def compress_text_part(text: str, target_size: int, text_type: str, model_name="Qwen/Qwen2.5-72B-Instruct") -> str:
    """
    Сжать переданный текст до приблизительно target_size слов с акцентом на особенности text_type.
    """
    # Ключевые слова для разных типов текста
    keywords = {
        "художественный": "plot, dialogues, characters",
        "учебный": "definitions, principles, examples",
        "научно-популярный": "examples, application",
        "научный": "hypothesis, methods, conclusions"
    }
    kws = keywords.get(text_type.lower(), "")
    messages = [
        {
            "role": "user",
            "content": (
                f"Please summarize the following text to approximately {target_size} words, "
                f"paying special attention to {kws}. Do not add any additional commentary; "
                f"simply summarize the text. Write in Russian: {text}"
            )
        }
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=target_size * 3  # Дополнительные токены для избежания обрезания результата
    )
    return completion.choices[0].message.content


# ------------------------------
# Функция для сжатия произвольного текста с определением типа через систему анализа
def compress_text(text: str, final_target_size: int = 100, text_type: str = None) -> str:
    """
    Сжать текст до final_target_size слов.
    Если text_type не указан, то он определяется с использованием системы анализа.
    """
    # Если тип текста не задан, получить его через функцию анализа (локальный импорт)
    if text_type is None:
        from ai_tools.analyze_system import determine_text_type  # локальный импорт для избежания циклического импорта
        text_type = determine_text_type(text)

    # Разбиваем текст на предложения
    sentences = text.split('. ')
    # Добавляем точку в конец предложения, если её нет
    sentences = [s if s.endswith('.') else s + '.' for s in sentences if s.strip()]

    max_chunk_size = 700  # Максимальное число слов в чанке
    chunks = split_into_chunks(sentences, max_chunk_size)
    if not chunks:
        return text  # Если текст не удалось разбить – вернуть исходный текст

    # Определяем промежуточный target_size для каждого чанка
    intermediate_target_size = final_target_size // len(chunks)
    compressed_chunks = [
        compress_text_part(chunk, intermediate_target_size, text_type)
        for chunk in chunks
    ]
    combined_text = ' '.join(compressed_chunks)
    # Финальное сжатие объединённого текста до итогового размера
    final_summary = compress_text_part(combined_text, final_target_size, text_type)
    return final_summary



# ------------------------------
# Функция для сжатия текста книги по запросу пользователя с записью результата в файл
def compress_text_by_user_request(book_name: str, user_id: int, final_target_size: int, text_type: str = None) -> str:
    """
    Получить полный текст книги (например, из базы данных), сжать его и сохранить сжатый текст в файл.
    Если text_type не указан, он определяется автоматически через систему анализа.
    """
    from database.database import get_book_full_text  # локальный импорт для получения текста книги

    text = get_book_full_text(book_name, str(user_id))
    summary = compress_text(text, final_target_size, text_type)

    output_file_path = os.path.join(BOOKS_DIRECTORY, str(user_id), f"{book_name}_compressed.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(summary)

    print(f"Сжатый текст записан в файл: {output_file_path}")
    return output_file_path





