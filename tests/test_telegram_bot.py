from datetime import date

from conftest import make_chunk

from zeppelin_rag.telegram_bot import (
    DailyLimiter,
    format_answer,
    parse_allowed_users,
    split_message,
    to_telegram_html,
)


def test_parse_allowed_users():
    assert parse_allowed_users("") == frozenset()
    assert parse_allowed_users(None) == frozenset()
    assert parse_allowed_users("123, 456") == {123, 456}


def test_daily_limiter_counts_per_user_and_resets_next_day():
    limiter = DailyLimiter(limit=2)
    day = date(2026, 9, 29)

    assert limiter.allow(1, day) and limiter.allow(1, day)
    assert not limiter.allow(1, day)
    assert limiter.allow(2, day)
    assert limiter.allow(1, date(2026, 9, 30))


def test_to_telegram_html_escapes_and_keeps_bold():
    assert to_telegram_html("**Ответ:** 1 < 2 & [1]") == "<b>Ответ:</b> 1 &lt; 2 &amp; [1]"


def test_split_message_respects_limit_and_keeps_text():
    text = "\n".join(["а" * 30] * 10)

    parts = split_message(text, limit=100)

    assert all(len(p) <= 100 for p in parts)
    assert "\n".join(parts) == text


def test_split_message_cuts_a_single_huge_line():
    parts = split_message("б" * 250, limit=100)

    assert [len(p) for p in parts] == [100, 100, 50]


def test_format_answer_lists_sources():
    chunks = [make_chunk(1, "текст"), make_chunk(2, "текст")]

    (message,) = format_answer("Гелий легче воздуха [1].", chunks)

    assert message.startswith("Гелий легче воздуха [1].")
    assert "<i>Источники:</i>\n[1] Раздел 1\n[2] Раздел 2" in message
