import time
from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer
from src.utils import plot_pareto_front  # On importe votre nouvelle fonction

def main():
    print("==================================================")
    print("      PROJET MORS : Long Tail Recommendation      ")
    print("==================================================")

    # ---------------------------------------------------------
    # ÉTAPE 1 : Chargement et Préparation des Données
    # ---------------------------------------------------------
    # On initialise le loader (par défaut sur MovieLens)
    loader = DataLoader(processed_data_dir="data/processed", active_dataset='movielens')
    
    # Affichage du tableau récapitulatif (Table 2 de l'article)
    loader.display_table_2_summary()
    
    # Chargement des données réelles
    df = loader.load_active_dataset()
    titles = loader.load_item_titles() # Pour l'affichage des titres (optionnel)
    
    # Split Train/Test (80/20)
    train_df, _ = loader.get_train_test_split(df)
    
    # Calcul des stats (Moyenne/Variance) nécessaire pour le calcul de Nouveauté
    item_stats = loader.get_item_statistics(train_df)

    # ---------------------------------------------------------
    # ÉTAPE 2 : Filtrage Collaboratif (Phase 1)
    # ---------------------------------------------------------
    print("\n[PHASE 1] Entraînement du modèle CF (Item-Based)...")
    train_matrix = loader.get_user_item_matrix(train_df)
    
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    # Choix de l'utilisateur cible (ex: User 1)
    target_user = 1
    print(f"\n[PHASE 1] Génération du pool de candidats pour User {target_user}...")
    
    # On génère 100 candidats potentiels. L'optimiseur devra en choisir 10 parmi eux.
    candidates = cf.get_top_k_candidates(target_user, k=100)

    # ---------------------------------------------------------
    # ÉTAPE 3 : Optimisation Multi-Objectif (Phase 2)
    # ---------------------------------------------------------
    print(f"\n[PHASE 2] Démarrage de l'Optimiseur Évolutionnaire (MORS)...")
    
    # Initialisation de l'optimiseur
    # list_length=10 : On veut recommander 10 films à la fin
    # population_size=50 : On fait évoluer 50 listes en parallèle
    optimizer = MORSOptimizer(
        candidates=candidates, 
        item_stats=item_stats, 
        list_length=10, 
        population_size=50
    )
    
    start_time = time.time()
    # On lance l'évolution sur 100 générations
    solutions = optimizer.run(generations=100)
    elapsed = time.time() - start_time
    
    print(f"Optimisation terminée en {elapsed:.2f}s.")
    print(f"Nombre de solutions Pareto-Optimales trouvées : {len(solutions)}")

    # ---------------------------------------------------------
    # ÉTAPE 4 : Visualisation (Phase 3)
    # ---------------------------------------------------------
    print("\n[PHASE 3] Génération du graphique...")
    
    # Appel de la fonction qui est maintenant dans utils.py
    plot_pareto_front(solutions, target_user, save_path="pareto_front_user1.png")
    
    print("\n=== FIN DU PROGRAMME ===")
    print("Vérifiez le fichier 'pareto_front_user1.png' dans votre dossier.")

if __name__ == "__main__":
    main()