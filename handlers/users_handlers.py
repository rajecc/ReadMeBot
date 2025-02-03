import os
import hashlib
from aiogram import Router, F, Bot
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from config.config import Config, load_config
from lexicon.lexicon import LEXICON
from ai_tools.dataset import dataset
from keyboard_utils.user_keyboards import (
    start_keyboard,
    exclude_prev_keyboard,
    answer_length_keyboard,
    recommendation_mode_keyboard,
    ai_keyboard,
    create_pagination_keyboard,
    cancel_compress_keyboard,
    cancel_reading_button,
    create_mode1_history_keyboard,
    create_mode3_history_keyboard,
)
from database.database import get_book_full_text, load_users_db, save_users_db, get_total_pages, get_book_page, get_current_page
from ai_tools.summarize_system import compress_text_by_user_request
from ai_tools.analyze_system import ask_question
from ai_tools.recommendation_system import get_book_recommendations
# Путь к файлу базы данных пользователей
USERS_DB_PATH = "users_db.json"

# Загрузка конфигурации
config: Config = load_config('config/.env')
BOOKS_DIRECTORY = "books"
os.makedirs(BOOKS_DIRECTORY, exist_ok=True)

class UploadBookState(StatesGroup):
    waiting_for_book = State()

class ReadBookState(StatesGroup):
    reading = State()

class InputPrefsState(StatesGroup):
    waiting_for_input = State()
    waiting_for_input_mode3 = State()
    waiting_for_exclude_option = State()

class CompressBookState(StatesGroup):
    awaiting_daily_read_pages = State()
    awaiting_days_to_finish = State()

class ChattingWithModelState(StatesGroup):
    awaiting_message_for_model = State()

router = Router()

users_db = load_users_db()

@router.message(CommandStart())
async def process_start_command(message: Message, state: FSMContext):
    user_id = str(message.from_user.id)
    if user_id not in users_db:
        users_db[user_id] = {'books': [], 'reading_state': {}, 'recommendation_history': []}  # Добавляем нового пользователя в базу данных
        save_users_db(users_db)  # Сохраняем изменения в базу данных
        await message.answer("Добро пожаловать! Выберите действие.", reply_markup=start_keyboard)
    else:
        await message.answer("Добро пожаловать! Выберите действие.", reply_markup=start_keyboard)
    found_book_name = None
    for book in users_db[user_id]["reading_state"]:
        if users_db[user_id]["reading_state"][book]["is_session"]:
            found_book_name = book
            break
    if found_book_name:
        users_db[user_id]["reading_state"][found_book_name]["is_session"] = False
    await state.clear()

@router.callback_query(F.data == "upload")
async def process_upload_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Пожалуйста, отправьте файл книги в формате PDF или TXT.")
    await state.set_state(UploadBookState.waiting_for_book)
    await callback.answer()

@router.message(UploadBookState.waiting_for_book, F.document)
async def process_book_upload(message: Message, state: FSMContext, bot: Bot):
    file = message.document
    user_id = str(message.from_user.id)

    if not (file.mime_type in ["application/pdf", "text/plain"]):
        await message.answer("Формат файла не поддерживается. Пожалуйста, отправьте файл в формате PDF или TXT.")
        return

    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))
    os.makedirs(user_books_dir, exist_ok=True)

    file_path = os.path.join(user_books_dir, file.file_name)
    await bot.download(file, file_path)

    # Инициализация состояния чтения книги
    if file.file_name not in users_db[user_id]["reading_state"]:
        users_db[user_id]["reading_state"][file.file_name] = {
            "page": 0,
            "total_pages": 0
        }
    save_users_db(users_db)  # Сохранение изменений в файл

    await message.answer(f"Книга '{file.file_name}' успешно добавлена в ваш список.", reply_markup=start_keyboard)
    await state.clear()

@router.callback_query(F.data == "read")
async def process_read_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))

    if not os.path.exists(user_books_dir):
        os.makedirs(user_books_dir)

    books = [book for book in os.listdir(user_books_dir) if book != "books.txt"]

    if books:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(
                text=book,
                callback_data=f"read_book_{hashlib.md5(book.encode('utf-8')).hexdigest()}")]
            for book in books]
        )
        await callback.message.answer("Выберите книгу для чтения:", reply_markup=keyboard)
    else:
        await callback.message.answer("Ваш список книг пуст. Загрузите книгу с помощью кнопки 'Загрузить книгу'.")
    await callback.answer()

@router.callback_query(F.data.startswith("read_book_"))
async def process_book_selection(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    book_hash = callback.data.split("_")[2]

    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))
    books = [book for book in os.listdir(user_books_dir) if book != "books.txt"]

    found_book = None
    for book in books:
        book_hash_computed = hashlib.md5(book.encode('utf-8')).hexdigest()
        if book_hash_computed == book_hash:
            found_book = book
            break

    if found_book:
        book_name = found_book
        file_path = os.path.join(user_books_dir, book_name)

        total_pages = get_total_pages(user_books_dir, book_name)
        current_page = users_db[user_id]["reading_state"].get(book_name, {}).get("page", 0)
        reading_state = users_db[user_id]["reading_state"].get(book_name, {})

        # If the user has not opened the book before, ask for daily reading preferences
        if not reading_state.get("is_opened", False):
            await callback.message.answer(
                "Вы читаете эту книгу впервые. Сколько страниц в день вы готовы читать?",
                reply_markup=cancel_compress_keyboard
                )
            users_db[user_id]["reading_state"][book_name]["is_session"] = True
            await state.set_state(CompressBookState.awaiting_daily_read_pages)
            await state.update_data(book_name=book_name, total_pages=total_pages, current_page=current_page)
            save_users_db("users_db.json")
        else:
            users_db[user_id]["reading_state"].setdefault(book_name, {})["total_pages"] = total_pages

            content = get_book_page(user_books_dir, book_name, current_page)

            await state.set_state(ReadBookState.reading)
            await state.update_data(book_name=book_name, page=current_page, total_pages=total_pages)

            users_db[user_id]["reading_state"][book_name]["is_session"] = True

            await callback.message.edit_text(
                content,
                reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
            )
            await callback.answer()
            save_users_db(users_db)
    else:
        await callback.message.answer("Не удалось найти книгу.")
        await callback.answer()
@router.message(CompressBookState.awaiting_daily_read_pages)
async def handle_daily_read_pages(message: Message, state: FSMContext):
    try:
        daily_pages = int(message.text)
        data = await state.get_data()

        await message.answer(f"Сколько дней у вас есть на прочтение книги?", reply_markup=cancel_compress_keyboard)
        await state.set_state(CompressBookState.awaiting_days_to_finish)

        await state.update_data(daily_pages=daily_pages)
    except ValueError:
        await message.answer("Пожалуйста, введите число.")


@router.message(CompressBookState.awaiting_days_to_finish)
async def handle_days_to_finish(message: Message, state: FSMContext):
    try:
        days_to_finish = int(message.text)
        user_id = str(message.from_user.id)
        data = await state.get_data()
        book_name = data["book_name"]
        pages = data["daily_pages"]

        # Set the book as opened
        users_db[user_id]["reading_state"].setdefault(book_name, {})["is_opened"] = True

        await message.answer(
            f"Отлично! Мы учли ваши предпочтения: {data['daily_pages']} страниц в день за {days_to_finish} дней."
        )

        target_length = days_to_finish * pages * 500 // 6
        compress_text_by_user_request(book_name, int(user_id), target_length)

        user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))
        current_page = data['current_page']
        total_pages = data['total_pages']
        content = get_book_page(user_books_dir, book_name, current_page)

        await state.update_data(page=current_page)

        await message.answer(
            content,
            reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
        )

        await state.clear()
    except ValueError:
        await message.answer("Пожалуйста, введите число.")

@router.callback_query(F.data == "cancel_reading")
async def cancel_reading(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    data = await state.get_data()
    book_name = data.get("book_name", "")
    await callback.message.edit_text(
        LEXICON["/start"],
        reply_markup=start_keyboard
    )
    users_db[user_id]["reading_state"][book_name]["is_session"] = False
    save_users_db(users_db)
    await state.clear()

@router.callback_query(F.data == "cancel_compress")
async def cancel_compress(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    data = await state.get_data()
    book_name = data.get("book_name", "")
    users_db[user_id]["reading_state"][book_name]["is_opened"] = True
    save_users_db(users_db)

    user_id = str(callback.from_user.id)
    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))

    current_page = users_db[user_id]["reading_state"].get(book_name, {}).get("page", 0)
    total_pages = get_total_pages(user_books_dir, book_name)
    content = get_book_page(user_books_dir, book_name, current_page)
    save_users_db(users_db)

    await callback.message.edit_text(
        content,
        reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
    )
    await state.set_state(ReadBookState.reading)
    await state.update_data(book_name=book_name, page=current_page, total_pages=total_pages)
    await callback.answer()


@router.callback_query(F.data == 'forward')
async def process_forward_press(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))
    user_data = await state.get_data()
    book_name = user_data['book_name']  # Имя книги
    current_page = user_data['page']
    total_pages = get_total_pages(user_books_dir, book_name)

    if current_page % 50 == 0 and users_db[user_id]["reading_state"][book_name]["update_page"] < current_page:
        update_book_content(user_id, book_name, current_page)

    if current_page + 1 < total_pages:
        current_page += 1
        content = get_book_page(user_books_dir, book_name, current_page)
        users_db[user_id]["reading_state"][book_name]["page"] = current_page

        # Обновляем только для текущей книги
        await state.update_data(page=current_page)

        await callback.message.edit_text(
            content,
            reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
        )
        save_users_db(users_db)
    else:
        update_book_content(user_id, book_name, current_page)
        await callback.message.edit_text("Вы достигли конца книги.", reply_markup=create_pagination_keyboard('backward', f'{total_pages}/{total_pages}', 'forward', 'chat_with_ai', cancel_reading_button))
    await callback.answer()

@router.callback_query(F.data == 'backward')
async def process_backward_press(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    user_data = await state.get_data()
    book_name = user_data['book_name']
    current_page = user_data["page"]  # Текущая страница из базы данных
    total_pages = user_data['total_pages']
    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))

    if current_page - 1 >= 0:
        current_page -= 1
        content = get_book_page(user_books_dir, book_name, current_page)
        users_db[user_id]["reading_state"][book_name]["page"] = current_page
        # Обновляем только для текущей книги
        await state.update_data(page=current_page)

        await callback.message.edit_text(
            content,
            reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
        )
        save_users_db(users_db)
    else:
        await callback.message.edit_text("Вы уже на первой странице.", reply_markup=create_pagination_keyboard('backward', f'{total_pages}/{total_pages}', 'forward'))
    await callback.answer()

@router.callback_query(F.data == 'chat_with_ai')
async def process_ai_chat_press(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    book_name = data.get("book_name", "")
    await callback.message.answer("Вы находитесь в чате с ИИ-ассистентом. Чтобы задать вопрос по произведению, просто отправьте его.", reply_markup=ai_keyboard)
    await state.set_state(ChattingWithModelState.awaiting_message_for_model)
    await state.update_data(book_name=book_name)

@router.message(ChattingWithModelState.awaiting_message_for_model)
async def handle_user_question(message: Message, state: FSMContext):
    await state.update_data(question=message.text)
    await message.answer("Выберите размер ответа", reply_markup=answer_length_keyboard)

@router.callback_query(F.data.startswith("answer_"))
async def handle_user_question(callback: CallbackQuery, state: FSMContext):
    size = callback.data[7:]
    data = await state.get_data()
    await callback.message.answer(ask_question(data['question'], size), reply_markup=ai_keyboard)

@router.callback_query(F.data == "leave_ai_chat")
async def process_ai_leave_press(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    found_book_name = None
    for book in users_db[user_id]["reading_state"]:
        if users_db[user_id]["reading_state"][book]["is_session"]:
            found_book_name = book
            break
    if found_book_name:
        current_page = users_db[user_id]["reading_state"][found_book_name]["page"]
        total_pages = users_db[user_id]["reading_state"][found_book_name]["total_pages"]
        user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))

        content = get_book_page(user_books_dir, found_book_name, current_page)
        users_db[user_id]["reading_state"][found_book_name]["page"] = current_page
        await state.set_state(ReadBookState.reading)

        await state.update_data(page=current_page, book_name=found_book_name, total_pages=total_pages)

        await callback.message.edit_text(
            content,
            reply_markup=create_pagination_keyboard('backward', f'{current_page + 1}/{total_pages}', 'forward', 'chat_with_ai')
        )
    else:
        await callback.message.edit_text("Нет активной книги для чтения.", reply_markup=create_pagination_keyboard('backward', '0/0', 'forward'))
        await callback.message.answer(LEXICON['/start'], reply_markup=start_keyboard)

@router.callback_query(F.data == "preferences")
async def process_preferences_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Исключать ранее рекомендованные произведения?", reply_markup=exclude_prev_keyboard)
    await state.set_state(InputPrefsState.waiting_for_exclude_option)

@router.message(InputPrefsState.waiting_for_exclude_option)
async def process_exclude_option_input(message: Message, state: FSMContext):
    await state.update_data(exclude_option=message.text.lower() == "да")
    await message.answer("Выберите режим учета предпочтений", reply_markup=recommendation_mode_keyboard)

@router.callback_query(F.data == "mode_1")
async def process_mode1_press(callback: CallbackQuery):
    await callback.message.answer("Выберите произведение из списка", reply_markup=create_mode1_history_keyboard(callback.from_user.id))

@router.callback_query(F.data.startswith("mode_1_"))
async def process_mode1_recommendation(callback: CallbackQuery, state: FSMContext):
    book_name = callback.data[7:]
    text = get_book_full_text(book_name, str(callback.from_user.id))
    data = await state.get_data()
    await callback.message.answer(get_book_recommendations("", text, 1, callback.from_user.id, data['exclude_option'], dataset))
    await state.clear()

@router.callback_query(F.data == "mode_2")
async def process_mode2_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Кратко опишите книгу, которую вы хотите прочитать")
    await state.set_state(InputPrefsState.waiting_for_input)

@router.message(InputPrefsState.waiting_for_input)
async def process_mode2_recommendation(message: Message, state: FSMContext):
    data = await state.get_data()
    await message.answer(get_book_recommendations(message.text, "", 2, message.from_user.id, data['exclude_option'], dataset))
    await state.clear()

@router.callback_query(F.data == "mode_3")
async def process_mode3_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Кратко опишите книгу, которую вы хотите прочитать")
    await state.set_state(InputPrefsState.waiting_for_input_mode3)

@router.message(InputPrefsState.waiting_for_input_mode3)
async def process_mode3_history(message: Message, state: FSMContext):
    await state.update_data(input=message.text)
    await message.answer("Выберите произведение из списка", reply_markup=create_mode3_history_keyboard(message.from_user.user_id))

@router.callback_query(F.data.startswith("mode_3_"))
async def process_mode3_recommendation(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    book_name = callback.data[7:]
    book_text = get_book_full_text(book_name, str(callback.from_user.id))
    await callback.message.answer(get_book_recommendations(data["input_text"], book_text, 2, callback.from_user.id, data['exclude_option'], dataset))
    await state.clear()



