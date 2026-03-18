import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Configuration ---
CSV_FILE = 'anemia_dataset_labeled.csv'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

PREPROCESS_FUNCTION = tf.keras.applications.mobilenet_v2.preprocess_input

# --- 2. Load and Split the Data ---
try:
    # Load the dataset from the CSV file
    df = pd.read_csv(CSV_FILE)
    print(f"Successfully loaded {CSV_FILE} with {len(df)} entries.")
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE}' was not found. Please ensure it's in the correct directory.")
    exit()

# Stratified split to ensure label proportions are maintained
# First split: 80% train, 20% temporary (for validation and test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,       # 20% of the data for validation and testing
    random_state=42,
    stratify=df['label'] # Ensures proportional representation of labels
)

# Second split: Split the 20% temporary set into 10% validation and 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,       # 50% of the temp_df (which is 10% of the original)
    random_state=42,
    stratify=temp_df['label']
)

print("\n--- Data Split Summary ---")
print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images:    {len(test_df)}\n")

print("--- Label Distribution in Each Set ---")
print("Training:\n", train_df['label'].value_counts(normalize=True))
print("\nValidation:\n", val_df['label'].value_counts(normalize=True))
print("\nTesting:\n", test_df['label'].value_counts(normalize=True))


# --- 3. Create Data Generators with Augmentation ---

# Training generator with data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=PREPROCESS_FUNCTION,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Adjust brightness
    fill_mode='nearest'
)

# Validation and Test generator (only preprocessing, no augmentation)
# It's crucial not to augment validation or test data
validation_test_datagen = ImageDataGenerator(
    preprocessing_function=PREPROCESS_FUNCTION
)


# --- 4. Flow Data from DataFrame ---

print("\n--- Creating Data Iterators ---")

# Training data iterator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # For two classes (Anemic/Non-Anemic)
    shuffle=True
)

# Validation data iterator
validation_generator = validation_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # No need to shuffle validation data
)

# Test data iterator
test_generator = validation_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Crucial to set shuffle=False for final evaluation
)

print("\nData generators created successfully!")
print("You can now use these generators to train your model, for example:")
print("model.fit(train_generator, validation_data=validation_generator, ...)")