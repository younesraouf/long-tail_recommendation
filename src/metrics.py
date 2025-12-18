import pandas as pd

class PaperMetrics:
    """
    Implémentation stricte des métriques de l'article (Section 4.4).
    """

    @staticmethod
    def calculate_precision_eq12(recommendation_list, user_id, test_df, rating_threshold=4.0):
        """
        Eq. 12: Precision = pu / k
        pu: nombre d'items PERTINENTS (présents dans le Probe Set ET aimés).
        
        rating_threshold: La note minimale pour considérer qu'un item est "aimé" (ex: 4.0).
        """
        k = len(recommendation_list)
        if k == 0: return 0.0

        # 1. On filtre le Test Set :
        # - C'est le bon utilisateur
        # - ET la note est supérieure ou égale au seuil (ex: >= 4 étoiles)
        liked_items_mask = (test_df['user_id'] == user_id) & (test_df['rating'] >= rating_threshold)
        
        # On récupère seulement les IDs des films AIMÉS
        user_liked_items = test_df[liked_items_mask]['item_id'].values
        
        # 2. On compte les hits (Intersection entre recommandation et films aimés)
        hits = sum([1 for item in recommendation_list if item in user_liked_items])
        
        return hits / k

    @staticmethod
    def calculate_novelty_eq13(recommendation_list, item_stats):
        """
        Eq. 13: Novelty = (1/mk) * sum(di)
        
        Dans l'article :
        - m = nombre d'utilisateurs (ici 1 pour l'explorateur interactif)
        - k = longueur de la liste
        - di = degree of item i (nombre de fois que l'item a été noté = Popularité)
        
        ATTENTION INTERPRÉTATION :
        L'article dit : "Recommendation algorithm with low novelty is prone to recommend novel items."
        Cela signifie que cette formule calcule en réalité la POPULARITÉ MOYENNE.
        
        Une valeur BASSE (Low Value) = Une GRANDE Nouveauté (High Novelty).
        """
        k = len(recommendation_list)
        if k == 0: return 0.0
        
        total_degree = 0
        for item in recommendation_list:
            if item in item_stats.index:
                # di = degree of item i (colonne 'popularity' calculée dans le loader)
                degree = item_stats.loc[item]['popularity']
                total_degree += degree
            else:
                # Si l'item est si rare qu'il n'est pas dans les stats (cas limite), popularité = 0
                total_degree += 0 
        
        # Retourne la popularité moyenne
        return total_degree / k