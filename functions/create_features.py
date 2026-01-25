def create_features(df):

    cogs = df["sales"] * (df["cogs_ratio"] / 100)
    sga = df["sales"] * (df["sga_ratio"] / 100)

    df["profitability_a"] = (df["sales"] - cogs - sga - df["interest"]) / df["capital"]

    return df
