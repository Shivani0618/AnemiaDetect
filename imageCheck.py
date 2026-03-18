import pandas as pd
from PIL import Image
import os

# The original CSV file with all filepaths
SOURCE_CSV = 'anemia_dataset_labeled.csv'
# The new, clean CSV that will be created
CLEANED_CSV = 'anemia_dataset_cleaned.csv'

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(SOURCE_CSV)
    print(f" Loaded {len(df)} records from '{SOURCE_CSV}'")
except FileNotFoundError:
    print(f" Error: Could not find '{SOURCE_CSV}'. Make sure it's in the same folder.")
    exit()

# --- 2. Verify every image file ---
problematic_files = []
valid_indices = []

print("\n🔍 Starting image verification... this may take a moment.")
# We loop through each row in the dataframe using its index
for index, row in df.iterrows():
    filepath = row['filepath']
    try:
        # First, quickly check if the file path even exists
        if not os.path.exists(filepath):
            problematic_files.append((filepath, "File not found at this path"))
            continue

        # Try to open and verify the image file
        with Image.open(filepath) as img:
            img.verify()  # This checks for corruption without loading the full image
        
        # If the code reaches here, the image is valid
        valid_indices.append(index)

    except Exception as e:
        # If any error occurs, we mark the file as problematic
        problematic_files.append((filepath, f"Error: {e}"))

# --- 3. Report the findings ---
print("\n--- Verification Report ---")
if not problematic_files:
    print(" Great news! All image files are valid and can be read.")
else:
    print(f"Found {len(problematic_files)} problematic image(s):")
    
    country_counts = {'India': 0, 'Italy': 0}
    
    for path, reason in problematic_files:
        print(f"  - Path: {path}\n    Reason: {reason}")
        if 'India' in path:
            country_counts['India'] += 1
        elif 'Italy' in path:
            country_counts['Italy'] += 1
            
    print("\n--- Summary of Problematic Images ---")
    print(f"**India:** {country_counts['India']} corrupted or missing image(s)")
    print(f"**Italy:** {country_counts['Italy']} corrupted or missing image(s)")

   
    # Create a new dataframe containing only the rows with valid images
    cleaned_df = df.loc[valid_indices]
    
    cleaned_df.to_csv(CLEANED_CSV, index=False)
    
    print(f"\n A new file '{CLEANED_CSV}' has been created with {len(cleaned_df)} valid records.")
    print("Use this new file in your training script.")
