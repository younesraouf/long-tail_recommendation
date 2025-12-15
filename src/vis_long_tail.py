import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader

def plot_long_tail(dataset_name='movielens'):
    """
    Génère le graphique de distribution "Long Tail" (Similaire à la Fig. 1 de l'article).
    """
    # 1. Chargement des données
    loader = DataLoader(active_dataset=dataset_name)
    try:
        df = loader.load_active_dataset()
    except FileNotFoundError:
        print(f"Erreur: Veuillez d'abord lancer 'python preprocess.py'")
        return

    print("Calcul de la distribution des notes...")

    # 2. Compter le nombre de notes pour chaque item
    # On groupe par 'item_id' et on compte la taille du groupe
    item_counts = df.groupby('item_id').size()

    # 3. Trier par popularité décroissante (du plus populaire au moins populaire)
    sorted_counts = item_counts.sort_values(ascending=False).values

    # 4. Création du graphique
    plt.figure(figsize=(12, 6))
    
    # Tracer la ligne bleue
    plt.plot(sorted_counts, color='blue', linewidth=2, label='Popularité des items')
    
    # Remplir sous la courbe pour l'esthétique
    plt.fill_between(range(len(sorted_counts)), sorted_counts, color='blue', alpha=0.1)

    # --- Annotations pour expliquer le problème ---
    
    # On marque arbitrairement le top 20% (Tête) vs le reste (Queue)
    cutoff = int(len(sorted_counts) * 0.2)
    plt.axvline(x=cutoff, color='red', linestyle='--', label='Séparation Head/Tail (Pareto 20/80)')
    
    plt.text(cutoff/2, max(sorted_counts)/2, 'HEAD\n(Items Populaires)', 
             horizontalalignment='center', color='darkred', fontweight='bold')
    
    plt.text(cutoff + (len(sorted_counts)-cutoff)/2, max(sorted_counts)/4, 'LONG TAIL\n(Items Rares)', 
             horizontalalignment='center', color='darkgreen', fontweight='bold')

    # Titres et labels
    plt.title(f'Distribution Long Tail - {dataset_name.capitalize()}', fontsize=14)
    plt.xlabel('Items (Triés par popularité)', fontsize=12)
    plt.ylabel('Nombre de notes reçues', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarder ou afficher
    output_file = f"long_tail_{dataset_name}.png"
    plt.savefig(output_file)
    print(f"Graphique sauvegardé sous : {output_file}")
    plt.show()

if __name__ == "__main__":
    # Vous pouvez changer 'movielens' par 'jester' ou 'netflix'
    plot_long_tail('movielens')