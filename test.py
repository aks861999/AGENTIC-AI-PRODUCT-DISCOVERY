import pandas as pd

# ── CONFIG ── update these paths to your actual file locations
PRODUCTS_CSV    = "volumes/home_depot_data/product_descriptions.csv"   # product_uid, product_description
ATTRIBUTES_CSV  = "volumes/home_depot_data/attributes.csv"             # product_uid, name, value
OUT_PRODUCTS    = "volumes/home_depot_data/product_descriptions_50k.csv"
OUT_ATTRIBUTES  = "volumes/home_depot_data/attributes_50k.csv"
N               = 50_000



# ── STEP 1: Load products, get first 50k unique UIDs
print("Loading products...")
df_products = pd.read_csv(PRODUCTS_CSV)
print(f"  Total products loaded: {len(df_products):,}")

# Get the first 50,000 unique product_uids (preserving order)
top_50k_uids = df_products["product_uid"].drop_duplicates().iloc[:N]
uid_set = set(top_50k_uids)
print(f"  Unique UIDs selected: {len(uid_set):,}")

# Filter products table
df_products_filtered = df_products[df_products["product_uid"].isin(uid_set)]
df_products_filtered.to_csv(OUT_PRODUCTS, index=False)
print(f"  Saved: {OUT_PRODUCTS} → {len(df_products_filtered):,} rows")

# ── STEP 2: Load attributes, filter to same 50k UIDs
print("\nLoading attributes...")
df_attrs = pd.read_csv(ATTRIBUTES_CSV)
print(f"  Total attribute rows loaded: {len(df_attrs):,}")

df_attrs_filtered = df_attrs[df_attrs["product_uid"].isin(uid_set)]
df_attrs_filtered.to_csv(OUT_ATTRIBUTES, index=False)
print(f"  Saved: {OUT_ATTRIBUTES} → {len(df_attrs_filtered):,} rows")

# ── STEP 3: Sanity check
print("\n── Sanity Check ──")
print(f"  Products file UIDs: {df_products_filtered['product_uid'].nunique():,}")
print(f"  Attributes file UIDs: {df_attrs_filtered['product_uid'].nunique():,}")
print(f"  UID range: {top_50k_uids.min()} → {top_50k_uids.max()}")
print("Done ✅")