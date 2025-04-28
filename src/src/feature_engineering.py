def create_features(df):
    """Feature engineering."""
    df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['amount_log'] = df['transaction_amount'].apply(lambda x: np.log1p(x))
    df['is_location_mismatch'] = (df['merchant_location'] != df['user_location']).astype(int)
    return df
