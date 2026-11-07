import pandas as pd
import numpy as np
import json
from nlp_engine import get_text_embeddings, calculate_semantic_similarity
from matching_engine import HybridMatcher

# Initialize the custom gradient descent trainer
class CustomGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.array([0.5, 0.3, 0.2]) # Starting weights
        
    def fit(self, X, y):
        n_samples = len(y)
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights)
            error = y - y_pred
            # Calculus: Partial derivatives to find the direction of the error minimum
            gradients = -(2 / n_samples) * np.dot(X.T, error)
            # Update weights by taking a step down the gradient slope
            self.weights -= self.lr * gradients
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {np.mean(error ** 2):.4f}")
                
    def save_weights(self):
        # Serialize the optimized weights for production use
        w_dict = {"w_text": float(self.weights[0]), "w_mbti": float(self.weights[1]), "w_loc": float(self.weights[2])}
        with open("production_weights.json", 'w') as f:
            json.dump(w_dict, f, indent=4)
        print(f"\nFinal Weights Saved: {w_dict}")

if __name__ == "__main__":
    print("1. Loading datasets.")
    users_df = pd.read_csv('Datasets/users.csv').set_index('user_id')
    feedback_df = pd.read_csv('Datasets/feedback.csv')

    print("2. Generating NLP embeddings for all 100 users.")
    users_df['full_bio'] = users_df['professional_summary'] + " " + users_df['about_me']
    all_embeddings = get_text_embeddings(users_df['full_bio'].tolist())
    # Create a fast lookup dictionary for the embeddings
    emb_dict = {uid: emb for uid, emb in zip(users_df.index, all_embeddings)}

    print("3. Building the training matrix from the 779 feedback interactions.")
    matcher = HybridMatcher()
    X_data, y_data = [], []

    for _, row in feedback_df.iterrows():
        u1, u2 = row['user_id'], row['matched_user_id']
        
        # Calculate the 3 component scores for this specific historical match
        text_sim = calculate_semantic_similarity(emb_dict[u1], emb_dict[u2])
        mbti_score = matcher.get_mbti_score(users_df.loc[u1, 'mbti'], users_df.loc[u2, 'mbti'])
        loc_score = matcher.get_location_score(users_df.loc[u1, 'location'], users_df.loc[u2, 'location'])
        
        X_data.append([text_sim, mbti_score, loc_score])
        y_data.append(row['action'])

    X_matrix = np.array(X_data)
    y_vector = np.array(y_data)

    print("\n4. Initiating Gradient Descent Training.")
    trainer = CustomGradientDescent(learning_rate=0.05, epochs=1000)
    trainer.fit(X_matrix, y_vector)
    trainer.save_weights()