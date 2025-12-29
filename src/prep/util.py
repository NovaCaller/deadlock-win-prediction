import pandas as pd


def normalize_df(
    df: pd.DataFrame,
    features: list[str],
    normalization_params: dict[str, tuple[float, float]] = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    df = df.copy()

    if normalization_params is None:
        normalization_params = {}

    # find missing features
    missing_features = [f for f in features if f not in normalization_params]

    if missing_features:
        means = df[missing_features].mean()
        stds = df[missing_features].std(ddof=0)
        new_params = {f: (means[f], stds[f]) for f in missing_features}
        normalization_params = normalization_params | new_params

    # all features guaranteed to have params
    means_series = pd.Series({f: m for f, (m, _) in normalization_params.items() if f in features})
    stds_series  = pd.Series({f: s for f, (_, s) in normalization_params.items() if f in features})

    # normalization
    df[means_series.index] = (df[means_series.index] - means_series) / stds_series

    return df, normalization_params
