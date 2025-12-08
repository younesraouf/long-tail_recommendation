STDIN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    """
    Gère le Filtrage Collaboratif (Phase 2 du papier).
    """
    def __init__(self, train_matrix):
        self.train_matrix = train_matrix
        self.similarity_matrix = None

    def compute_similarity(self):
        """Calcule la similarité Cosine entre les items."""
        print("Calcul de la matrice de similarité...")
        # TODO: Implémenter cosine_similarity
        pass

    def get_top_k_candidates(self, user_id, k=50):
        """
        Retourne les k meilleurs items prédits pour un utilisateur.
        C'est le pool de départ pour l'algorithme génétique.
        """
        # TODO: Prédire les notes et trier
        pass
