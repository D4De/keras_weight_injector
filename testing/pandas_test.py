import argparse
import pandas as pd
import os


def compare_dataframes(df1, df2, ignore_float_precision=True, ignore_null_diff=True):
    """
    Compares two pandas DataFrames and returns a summary of differences,
    with options to ignore common differences between pandas and csv output.
    """
    # Copia i dataframe per non modificare gli originali
    df1_normalized = df1.copy()
    df2_normalized = df2.copy()
    
    # Normalizza numeri float che sono matematicamente interi
    if ignore_float_precision:
        for df in [df1_normalized, df2_normalized]:
            for col in df.select_dtypes(include=['float']).columns:
                mask = df[col].notna() & df[col].apply(lambda x: x.is_integer() if hasattr(x, 'is_integer') else False)
                df.loc[mask, col] = df.loc[mask, col].astype(int)
    
    # Normalizza valori nulli
    if ignore_null_diff:
        df1_normalized = df1_normalized.fillna('')
        df2_normalized = df2_normalized.fillna('')
    
    # Confronta dopo la normalizzazione
    if df1_normalized.equals(df2_normalized):
        return "DataFrames are semantically identical (after normalization)."
    
    differences = []
    
    # Il resto del codice è identico...
    # Check for different columns
    if set(df1_normalized.columns) != set(df2_normalized.columns):
        differences.append("Different columns found.")
        differences.append(f"Columns in df1: {df1_normalized.columns.tolist()}")
        differences.append(f"Columns in df2: {df2_normalized.columns.tolist()}")
    
    # Check for different rows
    merged = pd.merge(df1_normalized, df2_normalized, how='outer', indicator=True)
    diff_rows = merged[merged['_merge'] != 'both']
    
    if not diff_rows.empty:
        differences.append("Different rows found:")
        differences.append(diff_rows.to_string(index=False))
    
    return "\n".join(differences) if differences else "No differences found."

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    
    # Argomenti posizionali per i percorsi dei file
    parser.add_argument('--f1', type=str, help='Path of file1')
    parser.add_argument('--f2', type=str, help='Path of file2')
    
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    
    if not os.path.exists(args.f1):
        print(f"Error: file doesn't exists {args.f1}")
        return
    
    if not os.path.exists(args.f2):
        print(f"Error: file doesn't exists {args.f2}")
        return
    
    print(f" Comparing files {args.f1} / {args.f2}")
    
    df1 = pd.read_csv(args.f1)
    df2 = pd.read_csv(args.f2)
    
    result = compare_dataframes(df1, df2, ignore_float_precision=True, ignore_null_diff=True)
    

def test_null_normalization():
    """
    Verify that different types of null values (NaN, None, '')
    are correctly normalized and considered equal.
    """
    print("Running null value normalization test...")
    
    # Create two DataFrames with different types of null values
    import numpy as np
    
    # DataFrame 1: uses numpy's NaN
    df1 = pd.DataFrame({
        'col1': [1.0, 2.0, np.nan, 4.0],
        'col2': ['a', 'b', 'c', np.nan]
    })
    
    # DataFrame 2: uses Python's None and empty strings
    df2 = pd.DataFrame({
        'col1': [1, 2, None, 4],
        'col2': ['a', 'b', 'c', '']
    })
    
    # Compare without normalization (should be different)
    result_without_norm = compare_dataframes(df1, df2, ignore_float_precision=True, ignore_null_diff=False)
    
    # Compare with normalization (should be equal)
    result_with_norm = compare_dataframes(df1, df2, ignore_float_precision=True, ignore_null_diff=True)
    
    print("Without normalization:")
    print(result_without_norm)
    print("\nWith normalization:")
    print(result_with_norm)
    
    # Verify that normalization works
    assert "identical" in result_with_norm, "Null values were not correctly normalized"
    
    print("✅ Null value normalization test completed successfully")

if __name__ == "__main__":
    __main__()
    #test_null_normalization()