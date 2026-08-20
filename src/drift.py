import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.config import cfg
from src.telegram_utils import send_telegram

logger = logging.getLogger(__name__)

def detect_drift() -> dict:
    """
    Simple performance drift detection using recent prediction errors.
    """
    err_path = Path(cfg.paths.errors_file)
    if not err_path.exists():
        return {"status": "no_data"}

    df = pd.read_csv(err_path)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = datetime.now() - timedelta(days=cfg.drift.lookback_days)
    recent = df[df["date"] >= cutoff]

    if len(recent) < 5:
        return {"status": "insufficient"}

    mape = recent["abs_error_pct"].mean()
    dir_acc = recent["direction_correct"].mean() * 100 if "direction_correct" in recent.columns else 50

    drifted = False
    reasons = []
    if mape > cfg.drift.mape_threshold:
        drifted = True
        reasons.append(f"MAPE high: {mape:.2f}% > {cfg.drift.mape_threshold}%")
    if dir_acc < cfg.drift.dir_acc_threshold:
        drifted = True
        reasons.append(f"DirAcc low: {dir_acc:.1f}% < {cfg.drift.dir_acc_threshold}%")

    result = {
        "status": "drift" if drifted else "ok",
        "mape": round(mape, 3),
        "dir_acc": round(dir_acc, 1),
        "reasons": reasons,
        "n": len(recent)
    }

    # log
    drift_path = Path(cfg.paths.drift_file)
    drift_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mape": result["mape"],
        "dir_acc": result["dir_acc"],
        "status": result["status"]
    }])
    if drift_path.exists():
        row.to_csv(drift_path, mode="a", header=False, index=False)
    else:
        row.to_csv(drift_path, index=False)

    if drifted:
        msg = (
            f"⚠️ *Model Drift Detected*\n"
            f"MAPE: `{result['mape']}%`\n"
            f"Directional Acc: `{result['dir_acc']}%`\n"
            f"Reasons: {', '.join(reasons)}"
        )
        send_telegram(msg)
        logger.warning(f"DRIFT: {result}")

    return result
