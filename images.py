import os
import glob
import pandas as pd

# --- Configuration ---
# Assumes this script is in the root project folder, and the 'dataset' folder is in the same directory.
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_CSV = 'anemia_dataset_labeled.csv'
HGB_THRESHOLD = 11.0

def find_image_in_folder(patient_dir):
    """
    Finds the best image in a patient's folder, prioritizing '_palpebral' images.
    Searches for png, jpg, and jpeg files.
    """
    if not os.path.isdir(patient_dir):
        return None
    # Prioritize specific, segmented images if they exist
    for fname in sorted(os.listdir(patient_dir)):
        if '_palpebral' in fname.lower() and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(patient_dir, fname)
    # Fallback to the first available image if no specific one is found
    for fname in sorted(os.listdir(patient_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(patient_dir, fname)
    return None

def main():
    """
    Main script to process raw data, label images based on Excel files,
    and save the results to a single CSV file.
    """
    all_image_data = []
    print("Starting dataset processing...")

    for country in ['India', 'Italy']:
        # Correctly locate the Excel file inside the 'dataset' directory
        excel_file_path = os.path.join(RAW_DATA_DIR, f"{country}.xlsx")
        image_dir_path = os.path.join(RAW_DATA_DIR, country)

        if not os.path.exists(excel_file_path):
            print(f"Warning: Excel file not found at '{excel_file_path}'. Skipping this country.")
            continue

        try:
            df_clinical_data = pd.read_excel(excel_file_path)
        except Exception as e:
            print(f"Error reading {excel_file_path}: {e}")
            continue

        # Check for common column names for patient ID and Hemoglobin
        id_col = next((col for col in ['Number', 'Patient_ID', 'ID'] if col in df_clinical_data.columns), None)
        hgb_col = next((col for col in ['Hgb', 'HGB', 'Hb'] if col in df_clinical_data.columns), None)

        if id_col is None or hgb_col is None:
            print(f"Error: Could not find required patient ID or Hgb columns in {excel_file_path}. Skipping.")
            continue

        for _, row in df_clinical_data.iterrows():
            if pd.isna(row[id_col]) or pd.isna(row[hgb_col]):
                continue

            try:
                patient_id = str(int(row[id_col]))
                # Handle commas as decimal separators
                hgb_value = float(str(row[hgb_col]).replace(',', '.'))
                label = 'Anemic' if hgb_value < HGB_THRESHOLD else 'Non-Anemic'
            except (ValueError, TypeError):
                print(f"Warning: Skipping patient {row.get(id_col, 'N/A')} in {country} due to invalid data.")
                continue

            patient_folder_path = os.path.join(image_dir_path, patient_id)
            image_filepath = find_image_in_folder(patient_folder_path)

            if image_filepath:
                all_image_data.append({'filepath': image_filepath, 'label': label})
            else:
                print(f"Warning: No valid image found for patient ID '{patient_id}' in '{patient_folder_path}'.")

    if not all_image_data:
        print("Error: No image data was processed. Please check your folder structure and Excel files.")
        return

    # Create and display the final DataFrame
    final_df = pd.DataFrame(all_image_data)
    print("\nProcessing complete.")
    print("--- Final DataFrame Sample---")
    print(final_df.head())
    print("\n--- DataFrame Info ---")
    final_df.info()
    print("\n--- Label Distribution ---")
    print(final_df['label'].value_counts())

    # Save the final DataFrame to a CSV file
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataFrame successfully saved to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
