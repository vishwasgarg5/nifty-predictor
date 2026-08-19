import os
import requests
import logging

logger = logging.getLogger(__name__)

def send_telegram(message: str, parse_mode="Markdown"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("Telegram credentials missing")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False
