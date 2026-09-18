import pandas as pd
import numpy as np

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("STEP 1: READ THE CSV FILE")
print("="*80)
# Read CSV where header is on the 3rd row (index 2)
df_raw = pd.read_csv("sales_data.csv", header=2)
print("Initial DataFrame loaded (header=2):")
print(df_raw.head())

# Save shape before cleaning
initial_shape = df_raw.shape
print(f"\nInitial Dataset Shape: {initial_shape}")

print("\n" + "="*80)
print("STEP 2: RENAME COLUMNS")
print("="*80)
# Clean column names: remove leading/trailing whitespace, convert to lowercase, replace spaces with underscores
df = df_raw.copy()
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Cleaned Column Names:")
print(df.columns.tolist())

print("\n" + "="*80)
print("STEP 3: DROP UNNECESSARY ROWS")
print("="*80)
# Check for completely empty rows or rows with all NaN values (like row index 0 which had ',,,,,,,,,,,')
print("Checking for rows with all NaN / empty values...")
rows_all_null = df.isnull().all(axis=1)
print(f"Number of completely empty rows found: {rows_all_null.sum()}")

df = df.dropna(how='all').reset_index(drop=True)

# Also ensure numeric columns are converted to appropriate data types
numeric_cols = ['order_id', 'quantity', 'unit_price', 'sales', 'profit', 'discount']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nDataFrame after dropping empty top/extra rows & resetting index:")
print(df.head())

print("\n" + "="*80)
print("STEP 4: DROP COLUMNS (IF NEEDED)")
print("="*80)
print("Justification:")
print("- Column 'city': Dropped because 'city' has missing values and is non-essential for financial order & revenue calculations (focus of current analysis).")
df_dropped = df.drop(columns=['city'])
print("\nDataFrame after dropping 'city' column:")
print(df_dropped.head())
df = df_dropped

print("\n" + "="*80)
print("STEP 5: COUNT MISSING VALUES")
print("="*80)
missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts)

print("\n" + "="*80)
print("STEP 6: HANDLE MISSING VALUES")
print("="*80)
print("a. Dropping rows where 'sales' is missing...")
sales_missing_count = df['sales'].isnull().sum()
df = df.dropna(subset=['sales']).reset_index(drop=True)
print(f"Dropped {sales_missing_count} row(s) with missing 'sales'.")

print("\nb. Filling missing values in 'profit' (mean) and 'discount' (median)...")
profit_mean = df['profit'].mean()
discount_median = df['discount'].median()

print(f"Mean profit calculated: {profit_mean:.2f}")
print(f"Median discount calculated: {discount_median:.2f}")

df['profit'] = df['profit'].fillna(profit_mean)
df['discount'] = df['discount'].fillna(discount_median)

print("\nMissing values after Step 6:")
print(df[['sales', 'profit', 'discount']].isnull().sum())

print("\n" + "="*80)
print("STEP 7: HANDLE MISSING CATEGORICAL DATA")
print("="*80)
missing_cust_before = df['customer_name'].isnull().sum()
df['customer_name'] = df['customer_name'].fillna("Unknown")
print(f"Replaced {missing_cust_before} missing 'customer_name' value(s) with 'Unknown'.")

print("\n" + "="*80)
print("STEP 8: DETECT AND REMOVE DUPLICATES")
print("="*80)
duplicates_mask = df.duplicated(subset=['order_id'], keep=False)
duplicate_rows = df[duplicates_mask]
print("Duplicate rows based on 'order_id':")
print(duplicate_rows[['order_id', 'customer_name', 'category', 'product', 'sales']])

num_duplicates = df.duplicated(subset=['order_id'], keep='first').sum()
df = df.drop_duplicates(subset=['order_id'], keep='first').reset_index(drop=True)
print(f"\nRemoved {num_duplicates} duplicate row(s) based on 'order_id'.")

print("\n" + "="*80)
print("STEP 9: FILTERING AND CREATING NEW COLUMNS")
print("="*80)
print("Filter 1: All orders where unit_price > 20,000:")
high_unit_price = df[df['unit_price'] > 20000]
print(high_unit_price[['order_id', 'customer_name', 'product', 'unit_price']])

print("\nFilter 2: customer_name, category, and status for orders where unit_price > 10,000 and status == 'Completed':")
filter2 = df[(df['unit_price'] > 10000) & (df['status'] == 'Completed')]
print(filter2[['customer_name', 'category', 'status', 'unit_price']])

print("\nCreating new column 'total_amount': total_amount = quantity * unit_price - discount")
df['total_amount'] = df['quantity'] * df['unit_price'] - df['discount']

print("\nCreating new column 'customer_type': 'Bulk Buyer' if quantity >= 3 else 'Regular Buyer'")
df['customer_type'] = np.where(df['quantity'] >= 3, 'Bulk Buyer', 'Regular Buyer')

print("\nSample rows with new columns:")
print(df[['order_id', 'customer_name', 'quantity', 'unit_price', 'discount', 'total_amount', 'customer_type']].head())

print("\n" + "="*80)
print("STEP 10: FINAL CLEAN DATASET")
print("="*80)
final_shape = df.shape
print(f"Shape of dataset BEFORE cleaning: {initial_shape}")
print(f"Shape of dataset AFTER cleaning:  {final_shape}")

print("\nFirst 10 rows of cleaned dataset:")
print(df.head(10))
