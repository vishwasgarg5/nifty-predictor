
                "symbol"
            )

            if not symbol:

                continue

            # Only evaluate stocks with
            # trained models.
            if not model_store.exists(
                symbol
            ):

                logger.info(
                    "%s skipped: no saved model",
                    symbol,
                )

                continue

            metadata = (
                build_stock_features(
                    symbol
                )
            )

            quality = metadata.get(
                "quality",
                0.0,
            )

            if quality < minimum_quality:

                logger.info(
                    "%s rejected by feature "
                    "quality: %.2f < %.2f",
                    symbol,
                    quality,
                    minimum_quality,
                )

                continue

            enriched_candidates.append(

                {

                    "symbol": symbol,

                    "technical_score": float(
                        row.get(
                            "score",
                            0.0,
                        )
                    ),

                    "feature_quality": (
                        quality
                    ),

                    "features": (
                        metadata.get(
                            "features"
                        )
                    ),
                }
            )


        if not enriched_candidates:

            logger.warning(
                "No candidates passed feature quality."
            )

            send_telegram(
                f"⚠️ All candidates failed "
                f"feature quality for `{today}`"
            )

            return 0


        logger.info(
            "Feature quality survivors: %s",
            len(enriched_candidates),
        )


        # ====================================================
        # STEP 6 — PARALLEL CHALLENGER PREDICTIONS
        # ====================================================

        challenger_predictions, challenger_summaries = (
            run_challenger_predictions(
                enriched_candidates=enriched_candidates,
                prediction_date=today,
            )
        )

        for summary in challenger_summaries:
            logger.info(
                "Parallel model | model=%s | success=%s | predictions=%s | error=%s",
                summary.get("model_name"), summary.get("success"),
                summary.get("prediction_count"), summary.get("error"),
            )


        # ====================================================
        # STEP 7 — SAVED MODEL INFERENCE (CHAMPION)
        # ====================================================

        ml_candidates: list[
            dict
        ] = []

        for candidate in enriched_candidates:

            symbol = candidate[
                "symbol"
            ]

            prediction = (
                get_ml_prediction(

                    symbol=symbol,

                    store=model_store,
                )
            )

            if not prediction:

                continue

            candidate[
                "ml_prediction"
            ] = prediction

            candidate[
                "opportunity_score"
            ] = float(
                prediction.get(
                    "opportunity_score",
                    0.0,
                )
            )

            candidate[
                "confidence"
            ] = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

            base_score = (
                calculate_final_score(

                    technical_score=(
                        candidate[
                            "technical_score"
                        ]
                    ),

                    opportunity_score=(
                        candidate[
                            "opportunity_score"
                        ]
                    ),

                    confidence=(
                        candidate[
                            "confidence"
                        ]
                    ),

                    feature_quality=(
                        candidate[
                            "feature_quality"
                        ]
                    ),
                )
            )

            candidate[
                "final_score"
            ] = apply_market_regime_adjustment(

                score=base_score,

                prediction=prediction,

                regime=market_regime,
            )

            if not quality_gate(
                candidate,
                minimum_quality,
            ):

                logger.info(
                    "%s rejected by "
                    "quality gate",
                    symbol,
                )

                continue

            ml_candidates.append(
                candidate
            )


        if not ml_candidates:

            logger.warning(
                "No ML candidates survived."
            )

            send_telegram(
                f"⚠️ No ML candidates survived "
                f"quality gate for `{today}`"
            )

            return 0


        # ====================================================
        # STEP 7 — FINAL RANKING
        # ====================================================

        ml_candidates.sort(

            key=lambda item: item[
                "final_score"
            ],

            reverse=True,
        )

        top_n = get_top_n()

        final_candidates = (
            ml_candidates[:top_n]
        )

        logger.info(
            "FINAL TOP %s: %s",

            len(final_candidates),

            ", ".join(

                item["symbol"]

                for item
                in final_candidates
            ),
        )


        # ====================================================
        # STEP 8 — OHLC PREDICTIONS
        # ====================================================

        records: list[
            dict
        ] = []

        for candidate in final_candidates:

            symbol = candidate[
                "symbol"
            ]

            try:

                predictor = (
                    OHLCPredictor(
                        symbol
                    )
                )

                ohlc = (
                    predictor.predict_next()
                )

                if not ohlc:

                    logger.warning(
                        "No OHLC prediction for %s",
                        symbol,
                    )

                    continue

                ml = candidate[
                    "ml_prediction"
                ]

                record = {

                    "date": today,

                    "symbol": symbol,

                    # Traditional score
                    "technical_score": (
                        candidate[
                            "technical_score"
                        ]
                    ),

                    # Feature quality
                    "feature_quality": (
                        candidate[
                            "feature_quality"
                        ]
                    ),

                    # Multi-model outputs
                    "expected_return": (
                        ml.get(
                            "expected_return"
                        )
                    ),

                    "probability_up": (
                        ml.get(
                            "probability_up"
                        )
                    ),

                    "expected_risk": (
                        ml.get(
                            "expected_risk"
                        )
                    ),

                    "risk_adjusted_return": (
                        ml.get(
                            "risk_adjusted_return"
                        )
                    ),

                    "opportunity_score": (
                        ml.get(
                            "opportunity_score"
                        )
                    ),

                    "confidence": (
                        ml.get(
                            "confidence"
                        )
                    ),

                    "direction": (
                        ml.get(
                            "direction"
                        )
                    ),

                    "model_version": (
                        ml.get(
                            "model_version"
                        )
                    ),

                    "training_rows": (
                        ml.get(
                            "training_rows"
                        )
                    ),

                    "feature_version": (
                        ml.get(
                            "feature_version"
                        )
                    ),

                    "model_saved_at": (
                        ml.get(
                            "model_saved_at"
                        )
                    ),

                    "market_regime": (
                        market_regime.get(
                            "regime",
                            "UNKNOWN",
                        )
                    ),

                    "final_score": (
                        candidate[
                            "final_score"
                        ]
                    ),

                    # OHLC prediction
                    **ohlc,
                }

                records.append(
                    record
                )

            except Exception as error:

                logger.warning(
                    "OHLC prediction failed "
                    "for %s: %s",
                    symbol,
                    error,
                )


        if not records:

            logger.warning(
                "No final predictions created."
            )

            send_telegram(
                f"⚠️ No final predictions "
                f"for `{today}`"
            )

            return 0


        # ====================================================
        # STEP 9 — SAVE DAILY PREDICTIONS
        # ====================================================

        prediction_dir = Path(
            cfg.paths.predictions
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_file = (
            prediction_dir
            / f"{today}.csv"
        )

        pd.DataFrame(
            records
        ).to_csv(
            prediction_file,
            index=False,
        )

        logger.info(
            "Saved predictions: %s",
            prediction_file,
        )


        # ====================================================
        # STEP 10 — PREDICTION LEDGER
        # ====================================================

        ledger_records: list[
            dict
        ] = []

        for record in records:

            symbol = record[
                "symbol"
            ]

            valid, validation = (
                validate_prediction(
                    symbol,
                    record,
                )
            )

            if not valid:

                logger.warning(
                    "Prediction validation failed "
                    "for %s: %s",
                    symbol,
                    validation,
                )

                continue

            current_close = (
                get_current_close(
                    symbol
                )
            )

            ledger_records.append(

                {

                    "market_date": today,

                    "symbol": symbol,

                    "current_close": (
                        current_close
                    ),

                    "predicted_open": float(
                        record["Open"]
                    ),

                    "predicted_high": float(
                        record["High"]
                    ),

                    "predicted_low": float(
                        record["Low"]
                    ),

                    "predicted_close": float(
                        record["Close"]
                    ),

                    "expected_return": (
                        record.get(
                            "expected_return"
                        )
                    ),

                    "probability_up": (
                        record.get(
                            "probability_up"
                        )
                    ),

                    "expected_risk": (
                        record.get(
                            "expected_risk"
                        )
                    ),

                    "direction": (
                        record.get(
                            "direction"
                        )
                    ),

                    "confidence": (
                        record.get(
                            "confidence"
                        )
                    ),

                    "opportunity_score": (
                        record.get(
                            "opportunity_score"
                        )
                    ),

                    "final_score": (
                        record.get(
                            "final_score"
                        )
                    ),

                    "market_regime": (
                        record.get(
                            "market_regime"
                        )
                    ),

                    "data_quality_score": (
                        record.get(
                            "feature_quality"
                        )
                    ),

                    "feature_version": (
                        record.get(
                            "feature_version"
                        )
                    ),

                    "model_version": (
                        record.get(
                            "model_version"
                        )
                    ),

                    "model_name": "current",
                }
            )


        # Challenger rows are stored only for later evaluation.
        if challenger_predictions is not None and not challenger_predictions.empty:
            for _, challenger in challenger_predictions.iterrows():
                probability = challenger.get("direction_probability", 0.5)
                direction_value = challenger.get("predicted_direction")
                direction = "UP" if direction_value in (1, 1.0, "1") or (direction_value not in (-1, -1.0, "-1") and float(probability) >= 0.50) else "DOWN"
                ledger_records.append({
                    "market_date": today, "symbol": challenger.get("symbol"),
                    "current_close": None, "predicted_open": None,
                    "predicted_high": None, "predicted_low": None,
                    "predicted_close": None,
                    "expected_return": challenger.get("predicted_return"),
                    "probability_up": probability,
                    "expected_risk": challenger.get("predicted_risk"),
                    "direction": direction, "confidence": probability,
                    "opportunity_score": None, "final_score": None,
                    "market_regime": market_regime.get("regime", "UNKNOWN"),
                    "data_quality_score": None,
                    "feature_version": getattr(cfg.features, "feature_version", "v1"),
                    "model_version": challenger.get("model_name"),
                    "model_name": challenger.get("model_name"),
                })

        if ledger_records:

            record_predictions(
                ledger_records
            )

            logger.info(
                "Ledger records saved: %s",
                len(ledger_records),
            )


        # ====================================================
        # STEP 11 — NEWS SENTIMENT
        # ====================================================

        sentiments: dict = {}

        try:

            engine = (
                get_sentiment_engine()
            )

            max_articles = getattr(
                cfg.sentiment,
                "max_articles",
                10,
            )

            for record in records:

                symbol = record[
                    "symbol"
                ]

                try:

                    sentiments[symbol] = (
                        engine.analyze_stock(
                            symbol,
                            max_articles,
                        )
                    )

                except Exception as error:

                    logger.warning(
                        "Sentiment failed "
                        "for %s: %s",
                        symbol,
                        error,
                    )

        except Exception as error:

            logger.warning(
                "Sentiment engine failed: %s",
                error,
            )


        # ====================================================
        # STEP 12 — INDEX PREDICTIONS
        # ====================================================

        try:

            if (
                getattr(
                    cfg,
                    "indexes",
                    None,
                )
                and getattr(
                    cfg.indexes,
                    "enabled",
                    False,
                )
            ):

                predict_indexes()

        except Exception as error:

            logger.warning(
                "Index prediction failed: %s",
                error,
            )


        # ====================================================
        # STEP 13 — TELEGRAM REPORT
        # ====================================================

        lines = [

            "🚀 *AI STOCK PREDICTIONS*",

            (
                f"Date: `{today}` | "
                f"`{_uni_label()}`"
            ),

            "",

            (
                "*Market Regime:* "
                f"`{market_regime.get('regime', 'UNKNOWN')}`"
            ),

            "",

            "*TOP OPPORTUNITIES*",

            "",

            "```",

            (
                f"{'Stock':<11} "
                f"{'Dir':<5} "
                f"{'PUp':>5} "
                f"{'Ret%':>7} "
                f"{'Risk%':>7} "
                f"{'Score':>6}"
            ),

            "-" * 52,
        ]


        for record in records:

            symbol = str(
                record["symbol"]
            ).replace(
                ".NS",
                "",
            )

            direction = str(
                record.get(
                    "direction",
                    "N",
                )
            )

            probability_up = float(
                record.get(
                    "probability_up",
                    0.0,
                )
            )

            expected_return = float(
                record.get(
                    "expected_return",
                    0.0,
                )
            )

            expected_risk = float(
                record.get(
                    "expected_risk",
                    0.0,
                )
            )

            final_score = float(
                record.get(
                    "final_score",
                    0.0,
                )
            )

            lines.append(

                f"{symbol:<11} "
                f"{direction:<5} "
                f"{probability_up:>5.0%} "
                f"{expected_return:>+7.2%} "
                f"{expected_risk:>7.2%} "
                f"{final_score:>6.2f}"
            )


        lines.append(
            "```"
        )


        # ====================================================
        # OHLC SECTION
        # ====================================================

        lines += [

            "",

            "*OHLC PREDICTIONS*",

            "",

            "```",

            (
                f"{'Stock':<11} "
                f"{'Open':>9} "
                f"{'High':>9} "
                f"{'Low':>9} "
                f"{'Close':>9}"
            ),

            "-" * 52,
        ]


        for record in records:

            symbol = str(
                record["symbol"]
            ).replace(
                ".NS",
                "",
            )

            lines.append(

                f"{symbol:<11} "
                f"{float(record['Open']):>9.2f} "
                f"{float(record['High']):>9.2f} "
                f"{float(record['Low']):>9.2f} "
                f"{float(record['Close']):>9.2f}"
            )


        lines.append(
            "```"
        )


        # ====================================================
        # MODEL CONFIDENCE
        # ====================================================

        lines += [

            "",

            "*MODEL CONFIDENCE*",

        ]


        for record in records:

            symbol = str(
                record["symbol"]
            ).replace(
                ".NS",
                "",
            )

            confidence = float(
                record.get(
                    "confidence",
                    0.0,
                )
            )

            if confidence >= 0.75:

                emoji = "🟢"

            elif confidence >= 0.50:

                emoji = "🟡"

            else:

                emoji = "🔴"

            feature_quality = float(
                record.get(
                    "feature_quality",
                    0.0,
                )
            )

            lines.append(

                f"{emoji} {symbol}: "
                f"`{confidence:.0%}` | "
                f"Feature: `{feature_quality:.0%}`"
            )


        # ====================================================
        # SENTIMENT SECTION
        # ====================================================

        if sentiments:

            lines += [

                "",

                "*NEWS SENTIMENT*",

            ]

            for record in records:

                symbol = record[
                    "symbol"
                ]

                sentiment = sentiments.get(
                    symbol
                )

                if not sentiment:

                    continue

                try:

                    if not sentiment.article_count:

                        continue

                    score = float(
                        sentiment.overall_score
                    )

                    if score >= 0.15:

                        emoji = "🟢"

                    elif score <= -0.15:

                        emoji = "🔴"

                    else:

                        emoji = "⚪"

                    lines.append(

                        f"{emoji} "
                        f"{str(symbol).replace('.NS', '')}: "
                        f"`{score:+.2f}` "
                        f"({sentiment.overall_label})"
                    )

                except Exception:

                    continue


        # ====================================================
        # IPO WATCHLIST
        # ====================================================

        try:

            lines += (
                [""]
                + ipo_watchlist_telegram_lines()
            )

        except Exception as error:

            logger.warning(
                "IPO watchlist failed: %s",
                error,
            )


        # ====================================================
        # COMPLETION TIME
        # ====================================================

        elapsed = int(
            (
                datetime.now()
                - start
            ).total_seconds()
        )

        lines += [

            "",

            f"_Completed in {elapsed}s_",
        ]


        # ====================================================
        # SEND TELEGRAM
        # ====================================================

        send_telegram(
            "\n".join(
                lines
            )
        )


        logger.info(
            "=" * 70
        )

        logger.info(
            "PHASE 3C MORNING JOB COMPLETED"
        )

        logger.info(
            "=" * 70
        )

        return 0


    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        try:

            send_telegram(

                "❌ *Phase 3C Morning Job Failed*\n"

                f"Date: `{today}`\n"

                f"```{str(error)[:700]}```"
            )

        except Exception:

            pass

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
