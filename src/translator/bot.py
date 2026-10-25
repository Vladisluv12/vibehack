"""Telegram-бот: принимает видео / кружок / видео-файл, распознаёт жесты РЖЯ и отвечает текстом.

Запуск:
    pip install aiogram aiohttp-socks python-decouple
    в vibehack/.env (или через export):
        BOT_TOKEN=123456:ABC...
        TG_PROXY=socks5://127.0.0.1:12334     # если Telegram доступен только через VPN-клиент
    python bot.py --check     # проверить токен и VPN без модели
    python bot.py

Видео сохраняются в vibehack/uploads/<user_id>_<message_id>.mp4
(чтобы удалять их после распознавания: export KEEP_UPLOADS=0).
"""
import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, KeyboardButton, Message, ReplyKeyboardMarkup

from features import PROJECT, WORDS, NONE, MODEL_PATH
from recognizer import Recognizer, ru

try:  # .env через python-decouple, как в других ботах; без него — обычные переменные окружения
    from decouple import Config, RepositoryEnv
    _env = PROJECT / ".env"
    _cfg = Config(RepositoryEnv(str(_env))) if _env.exists() else None
except ImportError:
    _cfg = None


def env(key, default=None):
    if key in os.environ:
        return os.environ[key]
    if _cfg is not None:
        return _cfg(key, default=default)
    return default

UPLOADS = PROJECT / "uploads"
KEEP_UPLOADS = env("KEEP_UPLOADS", "1") != "0"
MAX_FILE_MB = 20          # больше Bot API скачать не даёт
# MediaPipe и модель грузят CPU — одновременно обрабатываем не больше N видео
sem = asyncio.Semaphore(int(env("MAX_PARALLEL", "1")))

dp = Dispatcher()
recognizer: Recognizer | None = None
_model_mtime = None


def get_recognizer():
    """Модель подхватывается сама: появилась после train.py или обновилась — загружаем заново.
    Бот можно запускать, пока модель ещё обучается."""
    global recognizer, _model_mtime
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if recognizer is None or mtime != _model_mtime:
        try:
            recognizer = Recognizer()
            _model_mtime = mtime
            logging.info("модель загружена: %s, классов %d", MODEL_PATH.name, len(recognizer.classes))
        except Exception:
            # train.py может как раз перезаписывать файл — попробуем при следующем видео
            logging.warning("не удалось загрузить модель, попробую позже", exc_info=True)
    return recognizer


BTN_HELP = "📖 Как пользоваться"
BTN_WORDS = "📝 Список слов"
BTN_STATUS = "🟢 Статус бота"
BTN_DONATE = "💛 Поддержать"
DONATE_CARD = "2202 2068 8532 1526"
KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_WORDS)],
        [KeyboardButton(text=BTN_STATUS), KeyboardButton(text=BTN_DONATE)],
    ],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Запиши кружок или пришли видео 🎥",
)

START = (
    "Привет! 👋 Я <b>Жестик</b> — переводчик с русского жестового языка в текст 🤟\n\n"
    "Запиши мне кружок 📹 или пришли видео 🎬 с жестами — и я переведу их в текст ✨\n\n"
    "Если что-то непонятно — загляни в /help 💡"
)
HELP = (
    "💡 <b>Как пользоваться</b>\n\n"
    "1️⃣ Коснись микрофона 🎤 справа от поля ввода — он сменится на камеру 📹. Удерживай его и показывай жесты: запишется кружок\n"
    "2️⃣ Или отправь готовое видео (mp4) — как видео или файлом 📎\n"
    "3️⃣ Подожди пару секунд ⏳ — я пришлю перевод текстом 💬\n\n"
    "📸 <b>Как снимать, чтобы я понял тебя лучше</b>\n"
    "• в кадре видны голова, плечи и обе руки целиком 🙆\n"
    "• хороший свет, камера не трясётся 💡\n"
    "• между жестами — короткая пауза, руки опущены ✋\n"
    "• видео не больше 20 МБ 📦\n\n"
    "📝 Какие слова я знаю — кнопка «Список слов» или /words"
)
DONATE_TEXT = (
    "💛 <b>Поддержать разработчика</b>\n\n"
    "Жестик бесплатный и делается в свободное время 🧑‍💻\n"
    "Если бот тебе помог или просто понравился — можно поддержать проект любой суммой 🙏\n\n"
    "💳 Перевод на карту Сбербанка:\n"
    f"<code>{DONATE_CARD}</code>\n"
    "<i>(нажми на номер, чтобы скопировать)</i>\n\n"
    "Каждый донат помогает учить Жестика новым словам 🌱\n"
    "Спасибо! 🤟"
)


def words_text():
    """Список слов берём из загруженной модели: словарь (vocab.json) мог уже вырасти, а модель — ещё нет."""
    rec = get_recognizer()
    ids = [w for w in (rec.classes if rec else WORDS) if w != NONE]
    names = sorted(ru(w) for w in ids)
    sep = "\n" if len(names) <= 40 else ", "      # длинный список — в строку, чтобы влез в сообщение
    body = sep.join(f"• {n}" for n in names) if sep == "\n" else sep.join(names)
    return f"📝 <b>Слова, которые я знаю</b> ({len(names)}):\n\n{body}\n\nСкоро выучу больше 🌱"

def format_result(r):
    lines = []
    if r.words:
        lines.append(f"💬 Перевод: <b>{r.text}</b>")
        if len(r.words) > 1 or r.words[0][1] < 0.8:
            lines.append("")
            lines.append("🔍 Уверенность:")
            lines += [f"{i}. {ru(w)} — {p:.0%}" for i, (w, p) in enumerate(r.words, 1)]
    else:
        lines.append("🤔 Не получилось уверенно распознать жест.")
        guesses = [(w, p) for w, p in r.candidates if w != NONE]
        if guesses:
            lines.append("Может быть: " + ", ".join(f"{ru(w)} ({p:.0%})" for w, p in guesses))
    if r.duration and r.hands_ratio < 0.3:
        lines.append("\n🙈 Руки почти не видны в кадре — попробуй отойти подальше от камеры.")
    elif not r.words:
        lines.append("\nПопробуй записать ещё раз — подсказки в /help 💡")
    return "\n".join(lines)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(START, reply_markup=KEYBOARD)


@dp.message(Command("help"))
@dp.message(F.text == BTN_HELP)
async def cmd_help(message: Message):
    await message.answer(HELP, reply_markup=KEYBOARD)


@dp.message(Command("words"))
@dp.message(F.text == BTN_WORDS)
async def cmd_words(message: Message):
    await message.answer(words_text(), reply_markup=KEYBOARD)


@dp.message(Command("donate"))
@dp.message(F.text == BTN_DONATE)
async def cmd_donate(message: Message):
    await message.answer(DONATE_TEXT, reply_markup=KEYBOARD)


@dp.message(Command("status"))
@dp.message(F.text == BTN_STATUS)
async def cmd_status(message: Message):
    rec = get_recognizer()
    if rec is None:
        text = ("🟢 Бот работает!\n\n"
                "🧠 Модель ещё учится — переводить пока не могу, "
                "но видео уже можно присылать: я их сохраню 💾")
    else:
        text = f"🟢 Бот работает!\n\n🧠 Модель готова, знаю {len(rec.classes) - 1} слов ✅"
    await message.answer(text, reply_markup=KEYBOARD)


@dp.message(F.video | F.video_note | (F.document & F.document.mime_type.startswith("video/")))
async def on_video(message: Message, bot: Bot):
    media = message.video or message.video_note or message.document
    if media.file_size and media.file_size > MAX_FILE_MB * 1024 * 1024:
        await message.reply(f"📦 Видео слишком большое (максимум {MAX_FILE_MB} МБ). Пришли покороче 🙏")
        return

    kind = "кружок" if message.video_note else "видео"
    status = await message.reply(f"📥 Получил {kind}, обрабатываю...")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{message.from_user.id}_{message.message_id}.mp4"
    keep = KEEP_UPLOADS
    try:
        await bot.download(media, destination=path)
        size_kb = path.stat().st_size // 1024
        duration = getattr(media, "duration", None)
        logging.info("user=%s сохранил %s (%d КБ, %s c)", message.from_user.id, path.name, size_kb, duration)

        rec = get_recognizer()
        if rec is None:
            keep = True  # распознаем потом, когда модель будет готова
            await status.edit_text(
                f"💾 {kind.capitalize()} сохранён ({size_kb} КБ"
                + (f", {duration} с" if duration else "") + ").\n\n"
                "🧠 Модель ещё учится, поэтому перевести пока не могу. Загляни чуть позже 🙏"
            )
            return

        await status.edit_text(f"🔎 Распознаю жесты... ⏳")
        async with sem:
            # распознавание блокирующее и тяжёлое — уводим в отдельный поток, чтобы бот не зависал
            result = await asyncio.to_thread(rec.recognize, path)
        logging.info("user=%s file=%s words=%s", message.from_user.id, path.name, result.words)
        await status.edit_text(format_result(result))
    except Exception:
        logging.exception("ошибка при обработке %s", path)
        await status.edit_text("😔 Не получилось обработать видео. Попробуй записать ещё раз.")
    finally:
        if not keep:
            path.unlink(missing_ok=True)


@dp.message()
async def other(message: Message):
    await message.answer("🎥 Пришли кружок или видео с жестами — и я переведу.\nЕсли что-то непонятно — /help 💡",
                         reply_markup=KEYBOARD)


DESCRIPTION = (
    "Перевожу русский жестовый язык в текст.\n\n"
    "Запиши кружок или пришли видео с жестами — я отвечу, что ты показал. "
    "Нажми «Старт», чтобы увидеть инструкцию и список слов."
)
SHORT_DESCRIPTION = "Перевожу русский жестовый язык в текст: пришли кружок с жестами"
COMMANDS = [
    BotCommand(command="start", description="Начать"),
    BotCommand(command="help", description="Как пользоваться"),
    BotCommand(command="words", description="Список слов, которые я знаю"),
    BotCommand(command="status", description="Работает ли бот и готова ли модель"),
    BotCommand(command="donate", description="Поддержать разработчика 💛"),
]


async def setup_profile(bot: Bot):
    """Описание и меню команд — вместо ручной настройки в @BotFather."""
    await bot.set_my_commands(COMMANDS)
    await bot.set_my_description(DESCRIPTION)
    await bot.set_my_short_description(SHORT_DESCRIPTION)


def make_bot():
    token = env("BOT_TOKEN")
    if not token:
        raise SystemExit("Не задан BOT_TOKEN: добавь его в vibehack/.env или export BOT_TOKEN=...")
    proxy = env("TG_PROXY")  # например socks5://127.0.0.1:12334 — локальный порт VPN-клиента
    session = AiohttpSession(proxy=proxy) if proxy else None
    logging.info("Telegram %s", f"через прокси {proxy}" if proxy else "напрямую (без прокси)")
    return Bot(token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


async def check():
    """python bot.py --check — проверить токен и VPN/прокси без модели."""
    bot = make_bot()
    try:
        me = await bot.get_me()
        print(f"OK: связь с Telegram есть, бот @{me.username} ({me.full_name})")
        await setup_profile(bot)
        print("OK: описание и меню команд обновлены")
        print("Модель:", "найдена" if MODEL_PATH.exists() else f"ещё нет ({MODEL_PATH})")
    except Exception as e:
        print(f"ОШИБКА: {type(e).__name__}: {e}")
        print("Проверь токен в .env, что VPN-клиент запущен и порт в TG_PROXY совпадает с его SOCKS-портом.")
    finally:
        await bot.session.close()


async def main():
    bot = make_bot()
    if get_recognizer() is None:
        logging.warning("модели %s пока нет — бот работает, видео сохраняет, "
                        "распознавание включится само после train.py", MODEL_PATH)
    await setup_profile(bot)
    await bot.delete_webhook(drop_pending_updates=True)  # не разбирать сообщения, пришедшие пока бот был выключен
    logging.info("бот запущен, жду видео")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(check() if "--check" in sys.argv else main())
