import os
import pandas as pd


def inspect_parquet_file(parquet_file_path):
    """Inspect the structure of a parquet file by loading it and printing columns and sample data"""

    if not os.path.exists(parquet_file_path):
        print(f"❌ File not found: {parquet_file_path}")
        return

    print(f"📁 Inspecting: {parquet_file_path}")

    try:
        # Read the parquet file
        df = pd.read_parquet(parquet_file_path)

        print(f"\n📊 File Info:")
        print(f"   • Rows: {len(df)}")
        print(f"   • Columns: {len(df.columns)}")
        print(f"   • File size: {os.path.getsize(parquet_file_path) / (1024*1024):.1f} MB")

        print(f"\n📋 Column Names:")
        for i, col in enumerate(df.columns):
            print(f"   {i+1:2d}. {col}")

        print(f"\n🔍 Column Data Types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")

        print(f"\n📝 Sample Data (First Row):")
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns:
                val = first_row[col]
                if pd.isna(val):
                    print(f"   {col}: NaN")
                elif isinstance(val, list):
                    print(f"   {col}: list with {len(val)} elements")
                elif isinstance(val, str):
                    if len(val) > 100:
                        print(f"   {col}: string (truncated): '{val[:100]}...'")
                    else:
                        print(f"   {col}: string: '{val}'")
                else:
                    print(f"   {col}: {type(val).__name__} = {val}")

    except Exception as e:
        print(f"❌ Error reading parquet file: {e}")
        import traceback

        traceback.print_exc()
