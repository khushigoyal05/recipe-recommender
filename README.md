# 1. Clone the repository
git clone https://github.com/khushigoyal05/recipe-recommender.git

# 2. Navigate into the project directory
cd recipe-recommender

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. IMPORTANT: Manually download 'RAW_recipes.csv' from Kaggle
#    (https://www.kaggle.com/datasets/realalexanderwei/food-com-recipes-with-ingredients-and-tags)
#    and place it inside a new 'data/' folder before proceeding.

# 5. Run the Streamlit application
streamlit run app.py