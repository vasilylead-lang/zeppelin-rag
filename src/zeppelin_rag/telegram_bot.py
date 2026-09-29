"""`zeppelin-bot` — Telegram front end for the RAG graph (long polling, no public URL needed).

Environment:
  TELEGRAM_BOT_TOKEN      bot token from @BotFather (required)
  TELEGRAM_ALLOWED_USERS  comma-separated Telegram user IDs; empty means everyone
  TELEGRAM_DAILY_LIMIT    questions per user per day (default 20); every question
                          spends Anthropic API credits
"""

import asyncio
import html
import logging
import os
import re
from datetime import date

from dotenv import load_dotenv

from zeppelin_rag.knowledge import Chunk

MAX_QUESTION_CHARS = 500
TELEGRAM_MESSAGE_LIMIT = 4096
# The graph is CPU- and API-bound; a couple of parallel runs is plenty for a small bot.
MAX_CONCURRENT_ANSWERS = 2

GREETING = (
    "Я отвечаю на вопросы о цеппелинах и дирижаблях по базе знаний: история, конструкция, "
    "аэродинамика, подъёмная сила, двигатели, катастрофы, современные дирижабли.\n\n"
    "Просто напишите вопрос. Например:\n"
    "• Почему гелий поднимает меньше водорода?\n"
    "• Зачем дирижаблю хвостовое оперение?\n"
    "• Сколько стоил билет на «Гинденбург»?"
)

log = logging.getLogger(__name__)


def parse_allowed_users(raw: str | None) -> frozenset[int]:
    return frozenset(int(part) for part in (raw or "").replace(" ", "").split(",") if part)


class DailyLimiter:
    """In-memory per-user daily quota; counters reset at midnight and on restart."""

    def __init__(self, limit: int):
        self.limit = limit
        self._day: date | None = None
        self._counts: dict[int, int] = {}

    def allow(self, user_id: int, today: date) -> bool:
        if today != self._day:
            self._day, self._counts = today, {}
        used = self._counts.get(user_id, 0)
        if used >= self.limit:
            return False
        self._counts[user_id] = used + 1
        return True


def to_telegram_html(text: str) -> str:
    """Escapes HTML and turns the model's Markdown **bold** into <b>bold</b>."""
    escaped = html.escape(text, quote=False)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Splits on paragraph, then line boundaries so every part fits one Telegram message."""
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(paragraph) > limit:
            parts.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        current = paragraph
    if current:
        parts.append(current)
    return parts


def format_answer(answer: str, chunks: list[Chunk]) -> list[str]:
    body = to_telegram_html(answer)
    if chunks:
        sources = "\n".join(
            f"[{i}] {html.escape(chunk.section, quote=False)}"
            for i, chunk in enumerate(chunks, start=1)
        )
        body += f"\n\n<i>Источники:</i>\n{sources}"
    return split_message(body)


def main() -> None:
    from telegram import Update
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    from zeppelin_rag.graph import build_default_graph

    load_dotenv()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level="INFO")
    # httpx logs every Telegram request URL, and the URL contains the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    allowed = parse_allowed_users(os.getenv("TELEGRAM_ALLOWED_USERS"))
    limiter = DailyLimiter(int(os.getenv("TELEGRAM_DAILY_LIMIT", "20")))

    log.info("Loading knowledge base and embedding model...")
    graph, _ = build_default_graph()
    slots = asyncio.Semaphore(MAX_CONCURRENT_ANSWERS)

    def is_allowed(update: Update) -> bool:
        return not allowed or update.effective_user.id in allowed

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_allowed(update):
            await update.message.reply_text("Этот бот работает только для приглашённых.")
            return
        await update.message.reply_text(GREETING)

    async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_allowed(update):
            await update.message.reply_text("Этот бот работает только для приглашённых.")
            return
        question = update.message.text.strip()
        if len(question) > MAX_QUESTION_CHARS:
            await update.message.reply_text(
                f"Слишком длинный вопрос: сократите его до {MAX_QUESTION_CHARS} символов."
            )
            return
        if not limiter.allow(update.effective_user.id, date.today()):
            await update.message.reply_text(
                f"Лимит — {limiter.limit} вопросов в сутки. Возвращайтесь завтра."
            )
            return

        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            async with slots:
                state = await asyncio.to_thread(graph.invoke, {"question": question, "steps": []})
        except Exception:
            log.exception("Graph failed for user %s", update.effective_user.id)
            await update.message.reply_text("Не удалось получить ответ. Попробуйте позже.")
            return

        for part in format_answer(state["answer"], state["documents"]):
            await update.message.reply_text(part, parse_mode=ParseMode.HTML)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
    log.info("Bot is polling for updates")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
