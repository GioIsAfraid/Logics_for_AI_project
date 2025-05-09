import numpy as np

def introduce_bias(df, bias_type, group_col, group_value, flip_prob, drop_frac, save_path):
    """
    Introduce bias into dataset and save to CSV.
    - For this case, we are eliminating 50% of female students who pass the test.
    """
    biased_df = df.copy()

    if bias_type == "drop":
        # Eliminate 50% of female students who passed the test
        condition = (biased_df[group_col] == group_value) & (biased_df["pass"] == 1)
        drop_indices = biased_df[condition].sample(frac=drop_frac, random_state=42).index
        biased_df = biased_df.drop(index=drop_indices)
    
    elif bias_type == "label_flip":
        # Flip labels (e.g., change some "pass" to "fail")
        condition = (biased_df[group_col] == group_value) & (biased_df["pass"] == 1)
        flip_mask = condition & (np.random.rand(len(biased_df)) < flip_prob)
        biased_df.loc[flip_mask, "pass"] = 0

    else:
        raise ValueError("bias_type must be 'label_flip' or 'drop'")

    # Save the biased dataframe to CSV
    biased_df.to_csv(save_path, index=False, sep=';')
    print(f"Biased dataset saved at: {save_path}")
