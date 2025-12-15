import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

class ItemBasedCF:
    """
    Implémentation du Filtrage Collaboratif (Item-Based).
    Gère la prédiction (Phase 1) et le calcul de Diversité (Evaluation).
    """

    def __init__(self, train_matrix):
        self.train_matrix = train_matrix
        self.similarity_matrix = None
        self.item_ids = train_matrix.columns.tolist()
        self.is_fitted = False

    def compute_similarity(self):
        """Calcule la matrice de similarité Cosine entre items."""
        print("[CF] Calcul de la similarité entre items (Cosine)...")
        start_time = time.time()
        
        # Transpose pour comparer les items (colonnes)
        item_features = self.train_matrix.T
        sim_matrix_np = cosine_similarity(item_features)
        
        self.similarity_matrix = pd.DataFrame(
            sim_matrix_np, 
            index=self.item_ids, 
            columns=self.item_ids
        )
        
        # Diagonale à 0
        np.fill_diagonal(self.similarity_matrix.values, 0)
        
        self.is_fitted = True
        elapsed = time.time() - start_time
        print(f"[CF] Matrice de similarité prête ({self.similarity_matrix.shape}). Temps: {elapsed:.2f}s")

    def predict_rating(self, user_id, item_id, k_neighbors=20):
        """Prédit la note pour un utilisateur et un item donné."""
        if not self.is_fitted:
            return 0.0

        if item_id not in self.similarity_matrix.index:
            return 0.0

        user_ratings = self.train_matrix.loc[user_id]
        rated_items_mask = user_ratings > 0
        rated_items = user_ratings[rated_items_mask].index
        
        if len(rated_items) == 0:
            return 0.0

        similarities = self.similarity_matrix.loc[item_id, rated_items]
        top_k = similarities.nlargest(k_neighbors)
        
        weighted_sum = 0.0
        sum_of_weights = 0.0
        
        for neighbor_item, similarity in top_k.items():
            if similarity > 0:
                rating = user_ratings[neighbor_item]
                weighted_sum += similarity * rating
                sum_of_weights += similarity
        
        if sum_of_weights == 0:
            return 0.0
            
        return weighted_sum / sum_of_weights

    def get_top_k_candidates(self, user_id, k=50):
        """Génère les K meilleurs items prédits pour un utilisateur."""
        user_vector = self.train_matrix.loc[user_id]
        unrated_items = user_vector[user_vector == 0].index.tolist()
        
        predictions = []
        for item in unrated_items:
            score = self.predict_rating(user_id, item)
            if score > 0:
                predictions.append((item, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]

    # --- C'EST ICI QUE L'INDENTATION ÉTAIT FAUSSE ---
    def calculate_list_diversity(self, item_list):
        """
        Calcule la Diversité Intra-Liste (Eq. 15).
        Plus c'est proche de 1, plus c'est diversifié.
        """
        k = len(item_list)
        if k < 2:
            return 0.0
            
        # On ne garde que les items connus dans la matrice
        valid_items = [i for i in item_list if i in self.similarity_matrix.index]
        
        if len(valid_items) < 2:
            return 0.0
            
        # Extraction de la sous-matrice de similarité
        sub_matrix = self.similarity_matrix.loc[valid_items, valid_items].values
        
        # Somme de toutes les similarités
        total_sim = np.sum(sub_matrix)
        
        # On retire la diagonale (sim=1) car la formule exclut a==b
        total_sim -= np.trace(sub_matrix)
        
        # Nombre de paires (k * (k-1))
        num_pairs = len(valid_items) * (len(valid_items) - 1)
        
        if num_pairs == 0: 
            return 0.0
            
        avg_similarity = total_sim / num_pairs
        
        # Diversité = 1 - Similarité Moyenne
        return 1.0 - avg_similarity