import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist

class MORSOptimizer:
    """
    Implémentation MOEA/D alignée sur le Tableau 1 de l'article.
    """

    def __init__(self, candidates: List[Tuple[int, float]], item_stats: pd.DataFrame, 
                 list_length: int = 10, population_size: int = 100, 
                 mutation_rate: float = 0.1,      # pm (Table 1)
                 crossover_rate: float = 0.9,     # pc (Table 1)
                 neighbor_size: int = 10,         # ns (Table 1)
                 update_size: int = 3):           # us (Table 1)
        
        self.candidates = candidates
        self.candidate_map = {item: score for item, score in candidates}
        self.item_stats = item_stats
        self.list_length = list_length
        self.pop_size = population_size
        
        # Paramètres avancés de l'article
        self.pm = mutation_rate
        self.pc = crossover_rate
        self.T = neighbor_size 
        self.nr = update_size 
        
        self.population = [] 
        self.weights = []
        self.neighborhoods = []
        self.z_ideal = [0.0, 0.0]
        self.graveyard = []

    # ... (Les méthodes evaluate, init_weights, init_neighborhoods, update_ideal_point 
    #      restent identiques au code précédent, je ne les répète pas pour abréger, 
    #      gardez celles que vous avez déjà) ...

    def _calculate_novelty_score(self, item_id: int) -> float:
        if item_id not in self.item_stats.index: return 0.0
        stats = self.item_stats.loc[item_id]
        mu = stats['mu']
        if mu <= 0.001: return 10.0 
        return 1.0 / (mu * (stats['sigma'] + 1.0)**2)

    def evaluate(self, individual: List[int]) -> List[float]:
        accuracy = sum([self.candidate_map.get(item, 0) for item in individual])
        novelty = sum([self._calculate_novelty_score(item) for item in individual])
        return [accuracy, novelty]

    def init_weights(self):
        self.weights = []
        for i in range(self.pop_size):
            w1 = i / (self.pop_size - 1) if self.pop_size > 1 else 1.0
            w2 = 1.0 - w1
            if w1 == 0: w1 = 0.0001
            if w2 == 0: w2 = 0.0001
            self.weights.append([w1, w2])
        self.weights = np.array(self.weights)

    def init_neighborhoods(self):
        dists = cdist(self.weights, self.weights)
        self.neighborhoods = []
        for i in range(self.pop_size):
            neighbors = np.argsort(dists[i])[:self.T]
            self.neighborhoods.append(neighbors)

    def update_ideal_point(self, f_values):
        if f_values[0] > self.z_ideal[0]: self.z_ideal[0] = f_values[0]
        if f_values[1] > self.z_ideal[1]: self.z_ideal[1] = f_values[1]

    def initialize_population(self):
        self.population = []
        self.graveyard = []
        self.z_ideal = [-1.0, -1.0]
        self.init_weights()
        self.init_neighborhoods()
        
        candidate_ids = [c[0] for c in self.candidates]
        if len(candidate_ids) < self.list_length:
            ind = candidate_ids
            self.population = [ind] * self.pop_size
            self.z_ideal = self.evaluate(ind)
            return

        for _ in range(self.pop_size):
            ind = random.sample(candidate_ids, self.list_length)
            self.population.append(ind)
            f = self.evaluate(ind)
            self.update_ideal_point(f)

    def tchebycheff_score(self, f_values, weight_idx):
        lambda_vec = self.weights[weight_idx]
        diff_acc = abs(self.z_ideal[0] - f_values[0])
        diff_nov = abs(self.z_ideal[1] - f_values[1])
        return max(lambda_vec[0] * diff_acc, lambda_vec[1] * diff_nov)

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        if len(parent1) < 2: return parent1
        point = random.randint(1, self.list_length - 1)
        child_raw = parent1[:point] + parent2[point:]
        return self._fix_duplicates(child_raw)

    def _fix_duplicates(self, individual: List[int]) -> List[int]:
        seen = set()
        clean_ind = []
        candidate_ids = [c[0] for c in self.candidates]
        for item in individual:
            if item not in seen:
                clean_ind.append(item)
                seen.add(item)
        while len(clean_ind) < self.list_length:
            new_item = random.choice(candidate_ids)
            if new_item not in seen:
                clean_ind.append(new_item)
                seen.add(new_item)
        return clean_ind

    def mutation(self, individual: List[int]) -> List[int]:
        # Utilisation de self.pm (Probability of Mutation)
        if random.random() < self.pm:
            idx = random.randint(0, self.list_length - 1)
            candidate_ids = [c[0] for c in self.candidates]
            new_gene = random.choice(candidate_ids)
            while new_gene in individual:
                new_gene = random.choice(candidate_ids)
            individual[idx] = new_gene
        return individual

    def run(self, generations: int = 50) -> List[Dict]:
        self.initialize_population()
        if not self.population: return []
        
        for gen in range(generations):
            for i in range(self.pop_size):
                neighbors_indices = self.neighborhoods[i]
                idx_p1 = np.random.choice(neighbors_indices)
                idx_p2 = np.random.choice(neighbors_indices)
                parent1 = self.population[idx_p1]
                parent2 = self.population[idx_p2]
                
                # --- Modification : Application de pc (Probability Crossover) ---
                if random.random() < self.pc:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1[:] # Pas de croisement, on clone le parent
                
                child = self.mutation(child)
                f_child = self.evaluate(child)
                
                self.update_ideal_point(f_child)
                
                shuffled_neighbors = np.random.permutation(neighbors_indices)
                replacement_count = 0
                
                for j in shuffled_neighbors:
                    # --- Modification : Utilisation de self.nr (Update Size 'us') ---
                    if replacement_count >= self.nr: break
                        
                    gte_child = self.tchebycheff_score(f_child, j)
                    neighbor_sol = self.population[j]
                    f_neighbor = self.evaluate(neighbor_sol)
                    gte_neighbor = self.tchebycheff_score(f_neighbor, j)
                    
                    if gte_child <= gte_neighbor:
                        self.graveyard.append({'items': neighbor_sol, 'accuracy': f_neighbor[0], 'novelty': f_neighbor[1]})
                        self.population[j] = child
                        replacement_count += 1

        # Constitution de la liste finale
        final_solutions = []
        seen_hashes = set()
        
        for ind in self.population:
            h = tuple(sorted(ind))
            if h not in seen_hashes:
                seen_hashes.add(h)
                scores = self.evaluate(ind)
                final_solutions.append({'items': ind, 'accuracy': scores[0], 'novelty': scores[1]})
        
        for dead_sol in reversed(self.graveyard):
            if len(final_solutions) >= self.pop_size:
                break
            h = tuple(sorted(dead_sol['items']))
            if h not in seen_hashes:
                seen_hashes.add(h)
                final_solutions.append(dead_sol)
        
        return final_solutions