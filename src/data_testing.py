import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

def test_no_missing_values(df: pd.DataFrame) -> None:
    total_missing = df.isnull().sum().sum()
    assert total_missing == 0, \
        f" TEST FAILED: {total_missing} missing values remain."
    print("PASS: no missing values")

def test_target_values (df: pd.DataFrame, target_col: str, allowed_values) -> None:  
    invalid_mask = ~df[target_col].isin(allowed_values)
    invalid_count = invalid_mask.sum()
    assert invalid_count == 0 , \
        f"TEST FAILED: {invalid_count} unexpected values in '{target_col}':" \
        f"{df.loc[invalid_mask, target_col].unique()}"
    print (f"PASS: '{ target_col }' contains only valid values.")

def compare_distributions (train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list ) -> dict:
    results = {}
    for col in columns:
        train_vals = train_df[col].dropna()
        test_vals = test_df [col].dropna()
        if train_df [col].dtype in ["int64", "float64"]:
            stat, p = ks_2samp(train_vals, test_vals)
            results [col] = {"test ": "KS", " statistic ": stat, "p_value": p }
        else:
            # Align categories
            train_counts = train_vals.value_counts(normalize = True)
            test_counts = test_vals.value_counts(normalize = True)
            all_cats = sorted (set(train_counts.index).union(test_counts.index))
            train_probs = [train_counts.get(c, 0) for c in all_cats]
            test_probs = [test_counts.get (c, 0) for c in all_cats]
            chi2 , p , _ , _ = chi2_contingency([train_probs, test_probs])
            results[col] = {"test": "Chi2", "statistic": chi2, "p_value": p}
    return results