import matplotlib.pyplot as plt
import os

def plot_pareto_front(solutions, user_id, save_path="pareto_front.png"):
    """
    Génère et sauvegarde le graphique du Front de Pareto.
    
    Axe X : Précision (Somme des notes prédites)
    Axe Y : Nouveauté (Score Long Tail)
    """
    print(f"[VISUALIZATION] Generating Pareto Front plot for User {user_id}...")
    
    accuracies = [sol['accuracy'] for sol in solutions]
    novelties = [sol['novelty'] for sol in solutions]
    
    plt.figure(figsize=(10, 6))
    
    # Tracer les points (Solutions)
    plt.scatter(accuracies, novelties, c='blue', alpha=0.6, edgecolors='k', s=80, label='Solutions MORS')
    
    # Mettre en évidence les extrêmes
    if len(accuracies) > 0:
        max_acc_idx = accuracies.index(max(accuracies))
        max_nov_idx = novelties.index(max(novelties))
        
        plt.scatter(accuracies[max_acc_idx], novelties[max_acc_idx], c='red', s=150, marker='*', label='Max Accuracy')
        plt.scatter(accuracies[max_nov_idx], novelties[max_nov_idx], c='green', s=150, marker='*', label='Max Novelty')
    
    # Décoration et Labels
    plt.title(f"MORS Pareto Front (User {user_id})\nTrade-off: Accuracy vs Novelty", fontsize=14)
    plt.xlabel("Total Predicted Accuracy (Sum of Ratings)", fontsize=12)
    plt.ylabel("Novelty Score (Long Tail)", fontsize=12)
    
    # Ajout d'une grille et légende
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Sauvegarde
    plt.savefig(save_path)
    print(f"[VISUALIZATION] Plot saved to '{save_path}'")
    plt.close() # Ferme la figure pour libérer la mémoire