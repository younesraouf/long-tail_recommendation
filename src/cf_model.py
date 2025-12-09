import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    """
    Item-Based Collaborative Filtering implementation.
    
    This class handles the prediction phase of the MORS framework.
    It computes item-item similarities and generates a pool of 
    candidate items based on predicted ratings.
    """

    def __init__(self, train_matrix):
        """
        Initializes the CF model.

        Args:
            train_matrix (pd.DataFrame): User-Item matrix (Rows=Users, Cols=Items).
        """
        self.train_matrix = train_matrix
        self.similarity_matrix = None
        self.item_ids = train_matrix.columns.tolist()

    def compute_similarity(self):
        """
        Computes the Cosine Similarity matrix between all items.
        
        The resulting matrix is symmetric (Item x Item).
        """
        print("Computing Cosine Similarity between items...")
        
        # Transpose because sklearn's cosine_similarity works on rows.
        # We need similarity between columns (Items), so we transpose.
        item_features = self.train_matrix.T
        
        # efficient calculation using sklearn
        sim_matrix_np = cosine_similarity(item_features)
        
        # Convert back to DataFrame to preserve Item IDs
        self.similarity_matrix = pd.DataFrame(
            sim_matrix_np, 
            index=self.item_ids, 
            columns=self.item_ids
        )
        print(f"Similarity matrix ready. Shape: {self.similarity_matrix.shape}")

    def predict_rating(self, user_id, item_id, k_neighbors=20):
        """
        Predicts the rating a user would give to a specific item.

        Formula: Weighted average of ratings from the k most similar items 
        that the user has already rated.

        Args:
            user_id (int): Target user ID.
            item_id (int): Target item ID.
            k_neighbors (int): Number of neighbors to consider.

        Returns:
            float: Predicted rating (or 0.0 if prediction is not possible).
        """
        # Safety check: if item is not in the matrix
        if item_id not in self.similarity_matrix.index:
            return 0.0

        # 1. Get user ratings
        user_ratings = self.train_matrix.loc[user_id]
        
        # 2. Filter for items actually rated by the user (> 0)
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return 0.0

        # 3. Get similarity between target item and rated items
        similarities = self.similarity_matrix.loc[item_id, rated_items]
        
        # 4. Keep top K neighbors
        top_k = similarities.nlargest(k_neighbors)
        
        # 5. Calculate weighted average
        weighted_sum = 0.0
        sum_of_weights = 0.0
        
        for neighbor_id, similarity in top_k.items():
            if similarity > 0: # Ignore negative correlations
                rating = user_ratings[neighbor_id]
                weighted_sum += similarity * rating
                sum_of_weights += similarity
        
        if sum_of_weights == 0:
            return 0.0
            
        return weighted_sum / sum_of_weights

    def get_top_k_candidates(self, user_id, k=50):
        """
        Generates the top-k recommendations for a specific user.
        
        These items serve as the "candidate pool" for the evolutionary algorithm.

        Args:
            user_id (int): Target user.
            k (int): Number of candidates to return.

        Returns:
            list: List of tuples [(item_id, predicted_rating), ...]
        """
        print(f"Generating candidates for User {user_id}...")
        
        # Identify items not yet rated by the user
        user_vector = self.train_matrix.loc[user_id]
        unrated_items = user_vector[user_vector == 0].index.tolist()
        
        predictions = []
        
        # Note: For larger datasets, this loop should be vectorized or parallelized.
        for item in unrated_items:
            score = self.predict_rating(user_id, item)
            if score > 0:
                predictions.append((item, score))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        top_candidates = predictions[:k]
        print(f"Generated {len(top_candidates)} candidates.")
        
        return top_candidates