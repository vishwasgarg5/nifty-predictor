import os, logging, requests
logger = logging.getLogger(__name__)

def send_telegram(message: str, parse_mode="Markdown") -> bool:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("Telegram secrets missing")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": parse_mode,
                  "disable_web_page_preview": True},
            timeout=15,
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(e)
        return False
