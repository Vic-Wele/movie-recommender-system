
# movie_recommender_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load data (this would be replaced with your actual data loading)
@st.cache_data
def load_data():
    # Load your preprocessed data here
    movies = pd.read_csv('data/processed/movies_final.csv')
    ratings = pd.read_csv('data/processed/ratings_final.csv')
    
    # Load mappings
    with open('data/processed/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    # Load predicted ratings matrix
    predicted_ratings = np.load('models/predicted_ratings_matrix.npy')
    
    return movies, ratings, mappings, predicted_ratings

# Recommendation functions (simplified versions)
def get_collaborative_recommendations(user_id, n_recommendations=10):
    """Get collaborative filtering recommendations"""
    try:
        user_idx = mappings['user_id_to_index'].get(user_id)
        if user_idx is None:
            return pd.DataFrame()
        
        user_predicted_ratings = predicted_ratings[user_idx, :]
        recommendations = []
        
        for movie_idx, pred_rating in enumerate(user_predicted_ratings):
            movie_id = mappings['index_to_movie_id'][movie_idx]
            actual_rating = user_item_matrix[user_idx, movie_idx]
            
            if actual_rating == 0:  # Only unrated movies
                recommendations.append({
                    'movie_id': movie_id,
                    'title': mappings['movie_id_to_title'].get(movie_id, 'Unknown'),
                    'predicted_rating': pred_rating
                })
        
        return pd.DataFrame(recommendations).sort_values('predicted_rating', ascending=False).head(n_recommendations)
    except:
        return pd.DataFrame()

def get_content_recommendations(movie_title, n_recommendations=5):
    """Get content-based recommendations"""
    try:
        movie_id = mappings['movie_title_to_id'].get(movie_title)
        if movie_id is None:
            return pd.DataFrame()
        
        # Load cosine similarity matrix (simplified)
        cosine_sim = np.load('data/processed/cosine_sim_matrix.npy')
        movie_idx = mappings['movie_id_to_index'][movie_id]
        sim_scores = list(enumerate(cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        recommendations = []
        for idx, score in sim_scores:
            movie_id = mappings['index_to_movie_id'][idx]
            recommendations.append({
                'title': mappings['movie_id_to_title'][movie_id],
                'similarity_score': score
            })
        
        return pd.DataFrame(recommendations)
    except:
        return pd.DataFrame()

def main():
    # Load data
    movies, ratings, mappings, predicted_ratings = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Recommendation Type",
        ["Collaborative Filtering", "Content-Based Filtering", "Hybrid Approach"]
    )
    
    # Main content based on selection
    if app_mode == "Collaborative Filtering":
        st.header("🤝 Collaborative Filtering Recommendations")
        st.write("Get personalized recommendations based on users with similar tastes.")
        
        user_id = st.number_input("Enter User ID", min_value=1, max_value=943, value=1)
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding recommendations..."):
                recommendations = get_collaborative_recommendations(user_id, n_recommendations)
                
            if not recommendations.empty:
                st.success(f"Top {n_recommendations} recommendations for User {user_id}:")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">#{i}: {row['title']}</div>
                            <div>Predicted Rating: {row['predicted_rating']:.2f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try a different user ID.")
    
    elif app_mode == "Content-Based Filtering":
        st.header("🎭 Content-Based Recommendations")
        st.write("Discover movies similar to ones you already like.")
        
        movie_title = st.selectbox(
            "Select a movie you like",
            sorted(movies['title'].dropna().unique())[:100]  # First 100 for performance
        )
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        
        if st.button("Find Similar Movies"):
            with st.spinner("Finding similar movies..."):
                recommendations = get_content_recommendations(movie_title, n_recommendations)
                
            if not recommendations.empty:
                st.success(f"Movies similar to '{movie_title}':")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">#{i}: {row['title']}</div>
                            <div>Similarity Score: {row['similarity_score']:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No similar movies found. Try a different movie.")
    
    else:  # Hybrid Approach
        st.header("🌟 Hybrid Recommendations")
        st.write("Get the best of both worlds with our hybrid approach.")
        
        user_id = st.number_input("Enter User ID", min_value=1, max_value=943, value=1, key="hybrid_user")
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10, key="hybrid_slider")
        
        if st.button("Get Hybrid Recommendations"):
            with st.spinner("Combining recommendations..."):
                # Simple hybrid approach - get both and combine
                collab_recs = get_collaborative_recommendations(user_id, n_recommendations)
                content_recs = get_content_recommendations("Star Wars (1977)", n_recommendations)  # Example
                
                # Combine results (simplified)
                hybrid_recs = pd.concat([collab_recs, content_recs], ignore_index=True)
                hybrid_recs = hybrid_recs.drop_duplicates('title').head(n_recommendations)
                
            if not hybrid_recs.empty:
                st.success(f"Hybrid recommendations for User {user_id}:")
                for i, (_, row) in enumerate(hybrid_recs.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">#{i}: {row['title']}</div>
                            <div>Score: {row.get('predicted_rating', row.get('similarity_score', 0)):.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No hybrid recommendations found.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Built with MovieLens 100K dataset using Collaborative Filtering and Content-Based approaches."
    )

if __name__ == "__main__":
    main()
