import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the pre-trained transformer model 
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embeddings(text_list):
    # Convert a list of strings into a matrix of 384-dimensional dense vectors
    return model.encode(text_list, show_progress_bar=True)

def calculate_semantic_similarity(vec1, vec2):
    # Reshape 1D arrays to 2D arrays (1 sample, n features) 
    v1_reshaped = vec1.reshape(1, -1)
    v2_reshaped = vec2.reshape(1, -1)
    
    # Compute the cosine angle and extract the raw scalar value
    score = cosine_similarity(v1_reshaped, v2_reshaped)[0][0]
    
    # Bound the score to prevent negative values 
    return max(0.0, float(score))

if __name__ == "__main__":
    # Test cases to validate the semantic logic
    bio_a = "Data analyst focused on predictive modeling and structured data pipelines."
    bio_b = "Machine learning engineer building classification algorithms and analytics tools."
    bio_c = "Creative director specializing in graphic design and frontend aesthetics."
    
    # Generate embeddings for the test cases
    embeddings = get_text_embeddings([bio_a, bio_b, bio_c])
    
    # Calculate similarity scores
    score_ab = calculate_semantic_similarity(embeddings[0], embeddings[1])
    score_ac = calculate_semantic_similarity(embeddings[0], embeddings[2])
    
    print(f"Similarity (Data vs ML): {score_ab:.4f}")
    print(f"Similarity (Data vs Design): {score_ac:.4f}")