from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.utils.keyboard import InlineKeyboardBuilder
from lexicon.lexicon import LEXICON
import os
import hashlib

# ------------------------------
# Основные кнопки
button_preferences = InlineKeyboardButton(text="✨ Предпочтения ✨", callback_data="preferences")
button_read = InlineKeyboardButton(text="📚 Читать 📖", callback_data="read")
button_upload = InlineKeyboardButton(text="⬆️ Загрузить 📥", callback_data="upload")
start_keyboard = InlineKeyboardMarkup(inline_keyboard=[[button_preferences], [button_read], [button_upload]])

# ------------------------------
# Кнопки выбора режима учета ранее рекомендованных книг
button_exclude_prev_yes = KeyboardButton(text="Да")
button_exclude_prev_no = KeyboardButton(text="Нет")
exclude_prev_keyboard = ReplyKeyboardMarkup(keyboard=[[button_exclude_prev_yes], [button_exclude_prev_no]])
remove_keyboard = ReplyKeyboardRemove()

# ------------------------------
# Кнопки выбора режима рекомендаций
button_mode_1 = InlineKeyboardButton(text="Только История", callback_data="mode_1")
button_mode_2 = InlineKeyboardButton(text="Только Ввод", callback_data="mode_2")
button_mode_3 = InlineKeyboardButton(text="Ввод + История", callback_data="mode_3")
recommendation_mode_keyboard = InlineKeyboardMarkup(inline_keyboard=[[button_mode_1], [button_mode_2], [button_mode_3]])

# ------------------------------
# Клавиатура для выбора длины ответа от AI
button_short_answer = InlineKeyboardButton(text="Краткий", callback_data="answer_short")
button_medium_answer = InlineKeyboardButton(text="Средний", callback_data="answer_medium")
button_detailed_answer = InlineKeyboardButton(text="Развернутый", callback_data="answer_detailed")
answer_length_keyboard = InlineKeyboardMarkup(inline_keyboard=[[button_short_answer, button_medium_answer, button_detailed_answer]])

# ------------------------------
# Кнопки управления чтением
cancel_reading_button = InlineKeyboardButton(text="Назад", callback_data="cancel_reading")
cancel_compress_button = InlineKeyboardButton(text="Назад", callback_data="cancel_compress")
cancel_compress_keyboard = InlineKeyboardMarkup(inline_keyboard=[[cancel_compress_button]])

# ------------------------------
# Клавиатура для чата с AI
ai_cancel_button = InlineKeyboardButton(text="Назад", callback_data="leave_ai_chat")
ai_keyboard = InlineKeyboardMarkup(inline_keyboard=[[ai_cancel_button]])

# ------------------------------
# Функция создания пагинации для чтения книги
def create_pagination_keyboard(*buttons: str) -> InlineKeyboardMarkup:
    kb_builder = InlineKeyboardBuilder()
    kb_builder.row(*[InlineKeyboardButton(
        text=LEXICON.get(button, button),
        callback_data=button) for button in buttons]
    )
    kb_builder.add(cancel_reading_button)
    return kb_builder.as_markup()

def create_mode1_history_keyboard(user_id) -> InlineKeyboardMarkup:
    user_id = str(user_id)
    user_books_dir = os.path.join("books", str(user_id))

    if not os.path.exists(user_books_dir):
        os.makedirs(user_books_dir)

    books = [book for book in os.listdir(user_books_dir) if book != "books.txt"]
    keyboard = InlineKeyboardMarkup

    if books:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(
                text=book,
                callback_data=f"mode1_{hashlib.md5(book.encode('utf-8')).hexdigest()}")]
                for book in books]
        )
    return keyboard

def create_mode3_history_keyboard(user_id) -> InlineKeyboardMarkup:
    user_id = str(user_id)
    user_books_dir = os.path.join("books", str(user_id))

    if not os.path.exists(user_books_dir):
        os.makedirs(user_books_dir)

    books = [book for book in os.listdir(user_books_dir) if book != "books.txt"]
    keyboard = InlineKeyboardMarkup

    if books:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(
                text=book,
                callback_data=f"mode3_{hashlib.md5(book.encode('utf-8')).hexdigest()}")]
                for book in books]
        )
    return keyboard



