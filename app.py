import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px;
    }
    .recipe-card {
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
        border-radius: 0.25rem;
    }
    .missing-ingredients {
        color: #f44336;
    }
    .match-score {
        font-weight: bold;
        color: #4CAF50;
    }
    .time {
        color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load and preprocess the dataset"""
    # Load the dataset (replace with your actual path)
    df = pd.read_csv('data/processed_recipes.csv')
    df['cleaned_ingredients'] = df['cleaned_ingredients'].apply(ast.literal_eval)
    df['ingredients_str'] = df['cleaned_ingredients'].apply(lambda x: ' '.join(x))
    return df

@st.cache_resource
def initialize_model(df):
    """Initialize the TF-IDF model"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])
    return vectorizer, tfidf_matrix

def clean_user_input(ingredients):
    """Clean and process user input"""
    lemmatizer = WordNetLemmatizer()
    cleaned = []
    
    for ingredient in ingredients:
        # Remove measurements
        ingredient = re.sub(r'[\d¼½¾⅓⅔⅛⅜⅝⅞]+[^a-zA-Z]*', '', ingredient)
        # Tokenize and lemmatize
        tokens = word_tokenize(ingredient.lower())
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove stop words and short words
        filtered = [word for word in lemmatized if len(word) > 2 and word not in ['and', 'or', 'of']]
        cleaned.append(' '.join(filtered))
    
    return cleaned

def find_missing_ingredients(recipe_ingredients, user_ingredients):
    """Find ingredients missing from user's list"""
    return [ing for ing in recipe_ingredients if ing not in user_ingredients]

def main():
    st.title("🍳 AI Recipe Recommender")
    st.markdown("Find recipes based on ingredients you have at home!")
    
    # Load data and initialize model
    df = load_data()
    vectorizer, tfidf_matrix = initialize_model(df)
    
    # User input
    ingredients_input = st.text_input(
        "Enter ingredients you have (comma separated):",
        placeholder="e.g., chicken, rice, tomatoes, onion"
    )
    
    if ingredients_input:
        # Process user input
        user_ingredients = [x.strip() for x in ingredients_input.split(',')]
        cleaned_input = clean_user_input(user_ingredients)
        processed_input = ' '.join(cleaned_input)
        
        # Vectorize user input
        input_vector = vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        df['similarity'] = cosine_similarities
        
        # Get top 5 recommendations
        recommendations = df.sort_values('similarity', ascending=False).head(5)
        
        # Display results
        st.subheader("Recommended Recipes")
        
        for idx, row in recommendations.iterrows():
            missing = find_missing_ingredients(row['cleaned_ingredients'], cleaned_input)
            
            with st.container():
                st.markdown(f"""
                <div class="recipe-card">
                    <h3>{row['name']}</h3>
                    <p><span class="match-score">Match: {row['similarity']*100:.1f}%</span> | 
                    <span class="time">Ready in: {row['minutes']} minutes</span></p>
                    {f'<p class="missing-ingredients">Missing ingredients: {", ".join(missing)}</p>' if missing else '<p>You have all ingredients!</p>'}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Recipe Instructions"):
                    st.markdown("**Ingredients:**")
                    st.write(", ".join(row['cleaned_ingredients']))
                    
                    st.markdown("**Instructions:**")
                    instructions = row['steps'].split('\n') if isinstance(row['steps'], str) else row['steps']
                    for i, step in enumerate(instructions, 1):
                        st.write(f"{i}. {step}")

if __name__ == "__main__":
    main()