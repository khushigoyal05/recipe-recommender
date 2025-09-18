import pandas as pd
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def clean_ingredients(ingredient_list):
    lemmatizer = WordNetLemmatizer()
    cleaned = []
    
    for ingredient in ingredient_list:
        # Remove measurements
        ingredient = re.sub(r'[\d¼½¾⅓⅔⅛⅜⅝⅞]+[^a-zA-Z]*', '', ingredient)
        # Tokenize and lemmatize
        tokens = word_tokenize(ingredient.lower())
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove stop words and short words
        filtered = [word for word in lemmatized if len(word) > 2 and word not in ['and', 'or', 'of']]
        cleaned.append(' '.join(filtered))
    
    return cleaned

def prepare_dataset():
    # Load the dataset
    data_path = os.path.join('data', 'RAW_recipes.csv')
    print(f"Loading data from {os.path.abspath(data_path)}")
    
    # First inspect the raw file
    with open(data_path, 'r', encoding='utf-8') as f:
        first_lines = [next(f) for _ in range(5)]
    print("First few lines of the file:")
    for line in first_lines:
        print(line.strip())
    
    # Try different ways to load the ingredients column
    try:
        # Attempt 1: Direct JSON load
        recipes_df = pd.read_csv(data_path)
        print("\nOriginal ingredients sample:")
        print(recipes_df['ingredients'].head(3))
        
        # Check if the column is already in list format
        if isinstance(recipes_df['ingredients'].iloc[0], str):
            print("\nAttempting to parse as JSON...")
            recipes_df['ingredients'] = recipes_df['ingredients'].apply(lambda x: json.loads(x))
        else:
            print("\nIngredients already in list format")
            
    except json.JSONDecodeError as e:
        print(f"\nJSON decode error: {e}")
        print("\nTrying alternative parsing method...")
        
        # Attempt 2: Manual parsing
        def parse_ingredients(ing_str):
            try:
                # Remove problematic characters
                ing_str = ing_str.replace("'", '"')
                return json.loads(ing_str)
            except:
                # Fallback for malformed JSON
                return [x.strip(" '") for x in ing_str.strip("[]").split(",")]
        
        recipes_df['ingredients'] = recipes_df['ingredients'].apply(parse_ingredients)
    
    # Clean ingredients for each recipe
    print("\nCleaning ingredients...")
    recipes_df['cleaned_ingredients'] = recipes_df['ingredients'].apply(clean_ingredients)
    
    # Create a simplified dataframe
    simplified_df = recipes_df[['id', 'name', 'cleaned_ingredients', 'minutes', 'n_steps', 'steps']]
    
    # Save the processed data
    output_path = os.path.join('data', 'processed_recipes.csv')
    simplified_df.to_csv(output_path, index=False)
    print(f"\nDataset prepared and saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    prepare_dataset()