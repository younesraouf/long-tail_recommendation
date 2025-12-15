import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import time

class MORSOptimizer:
    """
    Algorithme Évolutionnaire Multi-Objectif (MOEA) pour la Recommandation.
    
    Cette classe implémente la Phase 2 du framework MORS.
    Elle cherche le meilleur compromis (Pareto Front) entre :
    1. Précision (Maximiser la note prédite) - Eq. 11 (F1)
    2. Nouveauté (Maximiser l'inclusion d'items de la Long Tail) - Eq. 11 (F2) et Eq. 9
    """

    def __init__(self, candidates: List[Tuple[int, float]], item_stats: pd.DataFrame, 
                 list_length: int = 10, population_size: int = 50, mutation_rate: float = 0.3):
        """
        Initialise l'optimiseur.

        Args:
            candidates: Liste de tuples (item_id, predicted_rating) venant du CF.
            item_stats: DataFrame contenant 'mu' (moyenne) et 'sigma' (variance) pour chaque item.
            list_length: Nombre d'items à recommander (K).
            population_size: Taille de la population génétique.
            mutation_rate: Probabilité de mutation.
        """
        self.candidates = candidates
        # Dictionnaire rapide pour récupérer la note prédite d'un item {id: note}
        self.candidate_map = {item: score for item, score in candidates}
        
        self.item_stats = item_stats
        self.list_length = list_length
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []

    def _calculate_novelty_score(self, item_id: int) -> float:
        """
        Calcule le score de nouveauté pour un seul item.
        Basé sur l'Eq. (9) de Wang et al. (2016).
        
        Formule : m_i = 1 / (mu * (sigma + 1)^2)
        """
        if item_id not in self.item_stats.index:
            return 0.0
        
        stats = self.item_stats.loc[item_id]
        mu = stats['mu']
        sigma = stats['sigma']
        
        # Sécurité : Si un item n'a jamais été noté (mu=0), il est infiniment nouveau
        # On met une valeur plafond pour éviter la division par zéro
        if mu <= 0.001: 
            return 10.0 
        
        # Eq. 9
        return 1.0 / (mu * (sigma + 1.0)**2)

    def evaluate(self, individual: List[int]) -> Tuple[float, float]:
        """
        Évalue une solution (une liste de recommandations).
        
        Returns:
            tuple: (Score Précision F1, Score Nouveauté F2)
        """
        # Objectif 1: Précision (Somme des notes prédites) - Eq. 8
        accuracy = sum([self.candidate_map.get(item, 0) for item in individual])

        # Objectif 2: Nouveauté (Somme des scores de nouveauté) - Eq. 10
        novelty = sum([self._calculate_novelty_score(item) for item in individual])
        
        return accuracy, novelty

    def initialize_population(self):
        """Génère la population initiale aléatoirement depuis les candidats."""
        # print(f"[MORS] Initialisation de la population ({self.pop_size} individus)...")
        self.population = []
        candidate_ids = [c[0] for c in self.candidates]
        
        # Si on a moins de candidats que la longueur demandée, on ne peut rien optimiser
        if len(candidate_ids) < self.list_length:
            raise ValueError("Pas assez de candidats pour remplir la liste !")

        for _ in range(self.pop_size):
            # Échantillonnage aléatoire sans doublons
            ind = random.sample(candidate_ids, self.list_length)
            self.population.append(ind)

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Effectue un croisement (Single-Point Crossover).
        Coupe les parents en deux et échange les queues.
        """
        point = random.randint(1, self.list_length - 1)
        
        child1_raw = parent1[:point] + parent2[point:]
        child2_raw = parent2[:point] + parent1[point:]
        
        # Correction des doublons (un item ne peut pas être recommandé 2 fois)
        child1 = self._fix_duplicates(child1_raw)
        child2 = self._fix_duplicates(child2_raw)
        
        return child1, child2

    def _fix_duplicates(self, individual: List[int]) -> List[int]:
        """
        Supprime les doublons dans une liste et comble les trous avec de nouveaux items.
        """
        seen = set()
        clean_ind = []
        candidate_ids = [c[0] for c in self.candidates]
        
        # Garder les items uniques en préservant l'ordre
        for item in individual:
            if item not in seen:
                clean_ind.append(item)
                seen.add(item)
        
        # Remplir les trous avec des items aléatoires du pool qui ne sont pas déjà là
        while len(clean_ind) < self.list_length:
            new_item = random.choice(candidate_ids)
            if new_item not in seen:
                clean_ind.append(new_item)
                seen.add(new_item)
                
        return clean_ind

    def mutation(self, individual: List[int]) -> List[int]:
        """
        Effectue une mutation : Remplace un item au hasard par un autre du pool.
        """
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.list_length - 1)
            candidate_ids = [c[0] for c in self.candidates]
            
            new_gene = random.choice(candidate_ids)
            # S'assurer qu'on n'introduit pas un doublon
            while new_gene in individual:
                new_gene = random.choice(candidate_ids)
            
            individual[idx] = new_gene
            
        return individual

    def run(self, generations: int = 50) -> List[Dict]:
        """
        Exécute le processus évolutif.

        Args:
            generations (int): Nombre de générations.

        Returns:
            List[Dict]: Une liste de solutions uniques avec leurs scores.
        """
        self.initialize_population()
        
        for gen in range(generations):
            # 1. Évaluation
            scored_pop = [(ind, self.evaluate(ind)) for ind in self.population]
            
            # 2. Sélection (Approche simplifiée par somme pondérée)
            # L'article utilise MOEA/D (Decomposition). Ici on simule une pression vers un équilibre.
            # On trie par: Accuracy + (Novelty * 10) pour donner du poids à la nouveauté qui est souvent petite
            scored_pop.sort(key=lambda x: x[1][0] + x[1][1] * 20, reverse=True)
            
            # On garde le Top 50% comme parents (Elitisme)
            parents = [x[0] for x in scored_pop[:self.pop_size // 2]]
            
            next_generation = parents[:]
            
            # 3. Reproduction (Croisement + Mutation)
            while len(next_generation) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                
                c1, c2 = self.crossover(p1, p2)
                
                next_generation.append(self.mutation(c1))
                if len(next_generation) < self.pop_size:
                    next_generation.append(self.mutation(c2))
            
            self.population = next_generation
            
            # Log optionnel
            # if (gen + 1) % 10 == 0:
            #     best = self.evaluate(self.population[0])
            #     print(f"   [Gen {gen+1}] Best Acc: {best[0]:.2f} | Best Nov: {best[1]:.4f}")

        # Nettoyage final : On retourne les solutions uniques
        unique_solutions = {}
        for ind in self.population:
            key = tuple(sorted(ind)) # Tuple pour servir de clé de dictionnaire
            if key not in unique_solutions:
                acc, nov = self.evaluate(ind)
                unique_solutions[key] = {
                    'items': ind,
                    'accuracy': acc,
                    'novelty': nov
                }
        
        return list(unique_solutions.values())

# --- BLOC DE TEST RAPIDE ---
if __name__ == "__main__":
    from data_loader import DataLoader
    from cf_model import ItemBasedCF
    
    # 1. Préparation des données (Comme avant)
    loader = DataLoader(active_dataset='movielens')
    df = loader.load_active_dataset()
    titles = loader.load_item_titles()
    train_df, _ = loader.get_train_test_split(df)
    
    # Calcul des stats pour la Nouveauté (Phase 1b)
    item_stats = loader.get_item_statistics(train_df)
    
    # CF Model (Phase 1a)
    train_matrix = loader.get_user_item_matrix(train_df)
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    # Génération des candidats pour User 1
    target_user = 1
    candidates = cf.get_top_k_candidates(target_user, k=50)
    
    print("\n--- Démarrage de l'Optimiseur MORS ---")
    
    # 2. Initialisation de l'Optimiseur (Phase 2)
    optimizer = MORSOptimizer(candidates, item_stats, list_length=10, population_size=40)
    
    start_time = time.time()
    solutions = optimizer.run(generations=30)
    elapsed = time.time() - start_time
    
    print(f"Optimisation terminée en {elapsed:.2f}s.")
    print(f"Solutions trouvées : {len(solutions)}")
    
    # Affichage de 3 solutions représentatives (Extreme Précision, Extreme Nouveauté, Equilibré)
    # Tri par Précision
    solutions.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n[Solution 1] Focus PRÉCISION (Proche du CF classique) :")
    best_acc = solutions[0]
    print(f"   Acc: {best_acc['accuracy']:.2f} | Nov: {best_acc['novelty']:.4f}")
    print(f"   Films: {[titles.get(i, str(i)) for i in best_acc['items'][:3]]} ...")

    print("\n[Solution 2] Focus NOUVEAUTÉ (Long Tail) :")
    # Tri par Nouveauté
    solutions.sort(key=lambda x: x['novelty'], reverse=True)
    best_nov = solutions[0]
    print(f"   Acc: {best_nov['accuracy']:.2f} | Nov: {best_nov['novelty']:.4f}")
    print(f"   Films: {[titles.get(i, str(i)) for i in best_nov['items'][:3]]} ...")