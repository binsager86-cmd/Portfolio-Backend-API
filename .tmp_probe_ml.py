from app.services.eagle_eye.ml.feature_builder import build_events_from_ohlcv_cache, build_feature_matrix

rows = build_events_from_ohlcv_cache()
print('raw_rows', len(rows))
res = build_feature_matrix(rows)
print('after_rows', len(res.frame), 'rejected_total', sum(res.rejected_counts.values()))
if not res.frame.empty:
    counts = res.frame.groupby('ticker').size().sort_values(ascending=False)
    print('tickers', len(counts))
    print('top5', counts.head(5).to_dict())
    sectors = res.frame.groupby('sector_raw').size().sort_values(ascending=False)
    print('sectors', len(sectors))
    print('sector_counts', sectors.to_dict())
