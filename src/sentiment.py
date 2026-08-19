import logging
from typing import List, Dict
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gnews import GNews
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class StockSentiment:
    symbol: str
    overall_score: float
    overall_label: str
    headlines: list
    article_count: int
    method: str

class SentimentEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_ready = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.to(self.device).eval()
            self.finbert_ready = True
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e}. Will use VADER only.")
        
        self.vader = SentimentIntensityAnalyzer()
        self.gnews = GNews(language="en", country="IN", max_results=8, period="5d")

    def _finbert_score(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(**inputs).logits, dim=-1)[0]
        # 0=positive, 1=negative, 2=neutral
        return float(probs[0] - probs[1])

    def _vader_score(self, text: str) -> float:
        return self.vader.polarity_scores(text)["compound"]

    def analyze_stock(self, symbol: str, max_articles: int = 7) -> StockSentiment:
        headlines = []
        clean = symbol.replace(".NS", "")
        
        try:
            articles = self.gnews.get_news(f"{clean} stock")
            headlines = [a["title"] for a in articles if a.get("title")][:max_articles]
        except Exception:
            pass

        try:
            news = yf.Ticker(symbol).news or []
            for n in news[:4]:
                t = n.get("title")
                if t and t not in headlines:
                    headlines.append(t)
        except Exception:
            pass

        if not headlines:
            return StockSentiment(symbol, 0.0, "Neutral", [], 0, "none")

        scores = []
        method = "finbert" if self.finbert_ready else "vader"
        
        for h in headlines:
            try:
                if self.finbert_ready:
                    s = self._finbert_score(h)
                else:
                    s = self._vader_score(h)
                scores.append(s)
            except Exception:
                scores.append(self._vader_score(h))
                method = "vader_fallback"

        overall = sum(scores) / len(scores) if scores else 0.0
        label = "Bullish" if overall >= 0.15 else "Bearish" if overall <= -0.15 else "Neutral"

        return StockSentiment(
            symbol=symbol,
            overall_score=round(overall, 3),
            overall_label=label,
            headlines=list(zip(headlines, [round(s, 3) for s in scores])),
            article_count=len(headlines),
            method=method
        )

_engine = None
def get_sentiment_engine():
    global _engine
    if _engine is None:
        _engine = SentimentEngine()
    return _engine
