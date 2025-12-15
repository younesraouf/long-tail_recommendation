import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

class ItemBasedCF:
    """
    Implémentation du Filtrage Collaboratif basé sur les Items (Item-Based CF).
    
    Correspond à la Phase 1 du framework MORS (Section 2.2 et 3.1).
    Son but est de remplir la matrice de notes et de générer une liste "CF-K"
    d'items candidats pour chaque utilisateur.
    """

    def __init__(self, train_matrix):
        """
        Initialise le modèle CF.

        Args:
            train_matrix (pd.DataFrame): Matrice User-Item (Lignes=Users, Cols=Items).
                                         Les valeurs 0 indiquent une absence de note.
        """
        self.train_matrix = train_matrix
        self.similarity_matrix = None
        self.item_ids = train_matrix.columns.tolist()
        self.is_fitted = False

    def compute_similarity(self):
        """
        Calcule la matrice de similarité Cosine entre tous les items.
        
        Résultat : Une matrice symétrique (Item x Item) stockée dans self.similarity_matrix.
        """
        print("[CF] Calcul de la similarité entre items (Cosine)...")
        start_time = time.time()
        
        # On transpose car sklearn calcule la similarité entre les lignes.
        # Matrice originale : Users x Items
        # Transposée : Items x Users
        item_features = self.train_matrix.T
        
        # Utilisation de la version optimisée de sklearn (C-level)
        # sim_matrix_np est un numpy array
        sim_matrix_np = cosine_similarity(item_features)
        
        # On remet dans un DataFrame pour garder les ID des items
        self.similarity_matrix = pd.DataFrame(
            sim_matrix_np, 
            index=self.item_ids, 
            columns=self.item_ids
        )
        
        # On remplit la diagonale avec 0 pour qu'un item ne soit pas son propre voisin
        np.fill_diagonal(self.similarity_matrix.values, 0)
        
        self.is_fitted = True
        elapsed = time.time() - start_time
        print(f"[CF] Matrice de similarité prête ({self.similarity_matrix.shape}). Temps: {elapsed:.2f}s")

    def predict_rating(self, user_id, item_id, k_neighbors=20):
        """
        Prédit la note qu'un utilisateur donnerait à un item spécifique.
        
        Formule : Moyenne pondérée des notes des K items les plus similaires
        que l'utilisateur a déjà notés.
        """
        if not self.is_fitted:
            raise Exception("Le modèle n'est pas entraîné. Lancez compute_similarity() d'abord.")

        # Si l'item n'est pas dans la matrice (cas rare de split), on retourne 0
        if item_id not in self.similarity_matrix.index:
            return 0.0

        # 1. Récupérer toutes les notes de l'utilisateur
        # C'est une Series (index=item_id, value=rating)
        user_ratings = self.train_matrix.loc[user_id]
        
        # 2. On garde seulement les items qu'il a VRAIMENT notés (> 0)
        rated_items_mask = user_ratings > 0
        rated_items = user_ratings[rated_items_mask].index
        
        if len(rated_items) == 0:
            return 0.0

        # 3. Récupérer les similarités entre l'item cible et les items notés
        # On regarde la ligne 'item_id' dans la matrice de similarité
        similarities = self.similarity_matrix.loc[item_id, rated_items]
        
        # 4. Garder les Top-K voisins (les items les plus proches)
        top_k = similarities.nlargest(k_neighbors)
        
        # 5. Calcul de la moyenne pondérée
        weighted_sum = 0.0
        sum_of_weights = 0.0
        
        for neighbor_item, similarity in top_k.items():
            if similarity > 0: # On ignore les corrélations négatives ou nulles
                rating = user_ratings[neighbor_item]
                weighted_sum += similarity * rating
                sum_of_weights += similarity
        
        if sum_of_weights == 0:
            return 0.0
            
        return weighted_sum / sum_of_weights

    def get_top_k_candidates(self, user_id, k=50):
        """
        Génère la liste CF-K (Top-K items recommandés) pour un utilisateur.
        C'est cette liste qui est passée à l'optimiseur génétique ensuite.

        Args:
            user_id (int): ID de l'utilisateur cible.
            k (int): Longueur de la liste (ex: 50 ou 100).

        Returns:
            list: Liste de tuples [(item_id, predicted_rating), ...]
        """
        # print(f"[CF] Génération des candidats pour User {user_id}...")
        
        # Identifier les items NON notés par l'utilisateur (ceux qu'on peut recommander)
        user_vector = self.train_matrix.loc[user_id]
        unrated_items = user_vector[user_vector == 0].index.tolist()
        
        predictions = []
        
        # Note : Sur des gros datasets, on vectoriserait cette boucle.
        # Pour MovieLens 100k, la boucle simple suffit.
        for item in unrated_items:
            score = self.predict_rating(user_id, item)
            # On ne garde que les items qu'on pense qu'il aimera (ex: > 3.0)
            # Ou simplement tout ce qui est positif pour trier ensuite
            if score > 0:
                predictions.append((item, score))
        
        # Trier par note prédite décroissante (Les meilleures d'abord)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # On garde les K premiers
        top_candidates = predictions[:k]
        
        # print(f"   -> {len(top_candidates)} candidats trouvés (Top score: {top_candidates[0][1]:.2f})")
        return top_candidates

# --- BLOC DE TEST RAPIDE ---
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # 1. Charger les données
    loader = DataLoader(active_dataset='movielens')
    df = loader.load_active_dataset()
    
    # Charger les titres (Nouveau !)
    movie_titles = loader.load_item_titles()
    
    train_df, test_df = loader.get_train_test_split(df)
    train_matrix = loader.get_user_item_matrix(train_df)
    
    # 2. Initialiser le modèle CF
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    # 3. Test pour le User 1
    target_user = 1 # ID réel dans MovieLens
    
    print("\n--- Resultats pour l'Utilisateur 1 ---")
    
    # Générer la liste
    candidates = cf.get_top_k_candidates(target_user, k=10)
    
    print(f"\nTop 10 Recommandations (Precision seule) :")
    print(f"{'Rang':<5} | {'Note':<6} | {'ID':<5} | {'Titre du Film'}")
    print("-" * 60)
    
    for i, (item_id, score) in enumerate(candidates):
        # Récupérer le titre, ou mettre "Inconnu" si pas trouvé
        title = movie_titles.get(item_id, "Titre Inconnu")
        
        print(f"#{i+1:<4} | {score:.2f}   | {item_id:<5} | {title}")