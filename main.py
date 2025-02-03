import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from config.config import Config, load_config
from handlers import users_handlers
from database.database import load_users_db

print("Logger loading")
logger = logging.getLogger(__name__)


async def main():
    print("Main started")
    config: Config = load_config('config/.env')
    bot = Bot(token=config.tg_bot.token)
    storage = MemoryStorage()
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s:%(lineno)d #%(levelname)-8s '
               '[%(asctime)s] - %(name)s - %(message)s')

    logger.info('Starting bot')

    dp = Dispatcher(storage=storage)

    dp.include_router(users_handlers.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

    load_users_db()

asyncio.run(main())