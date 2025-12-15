import pandas as pd
import numpy as np
import os
import random

# Chemins des dossiers (basé sur votre structure)
RAW_DIR = '../data/raw'
PROCESSED_DIR = '../data/processed'

# Créer le dossier processed s'il n'existe pas
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_movielens():
    print("--- Traitement de MovieLens 100k ---")
    # Format: user id | item id | rating | timestamp
    # L'article utilise u.data
    file_path = os.path.join(RAW_DIR, 'ml-100k', 'u.data')
    
    # Chargement
    df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # On garde juste user, item, rating
    df = df[['user_id', 'item_id', 'rating']]
    
    # Sauvegarde
    output_path = os.path.join(PROCESSED_DIR, 'movielens_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"MovieLens sauvegardé : {df.shape[0]} notes. -> {output_path}")

def process_jester():
    print("\n--- Traitement de Jester (Dataset 1) ---")
    file_path = os.path.join(RAW_DIR, 'jester', 'jester-data-1.xls')
    
    print("Chargement du fichier Excel (cela peut prendre quelques secondes)...")
    # Chargement sans header (header=None) car le fichier brut n'en a pas
    df = pd.read_excel(file_path, header=None)
    
    # 1. Filtrage (comme demandé par l'article : users avec >= 36 votes)
    # La colonne 0 contient le nombre de ratings par utilisateur
    df_filtered = df[df[0] >= 36].copy()
    print(f"Utilisateurs après filtre (>=36 notes): {len(df_filtered)}")
    
    # On supprime la colonne compteur (0) maintenant qu'on a filtré
    df_filtered = df_filtered.drop(columns=[0])
    
    # 2. Nettoyage des données manquantes
    # 99.0 indique "pas de note" dans Jester. On remplace par NaN.
    df_filtered = df_filtered.replace(99.0, np.nan)
    
    # 3. Transformation en format long (User, Item, Rating)
    df_filtered['user_id'] = range(1, len(df_filtered) + 1)
    df_long = df_filtered.melt(id_vars=['user_id'], var_name='item_id', value_name='rating')
    
    # Supprimer les lignes vides (NaN)
    df_long = df_long.dropna()
    
    # 4. MAPPING DE L'ÉCHELLE [-10, 10] VERS [1, 5]
    # Formule de normalisation linéaire :
    # NewValue = ((OldValue - OldMin) / (OldMax - OldMin)) * (NewMax - NewMin) + NewMin
    # OldMin = -10, OldMax = 10 (Ecart de 20)
    # NewMin = 1, NewMax = 5 (Ecart de 4)
    
    print("Conversion de l'échelle -10/+10 vers 1-5...")
    
    # Formule simplifiée : ((x + 10) / 20) * 4 + 1
    df_long['rating'] = ((df_long['rating'] + 10) / 20) * 4 + 1
    
    # Arrondir si nécessaire (l'article dit "continuous", donc on garde les décimales, ex: 3.5)
    # Si vous voulez des entiers, décommentez la ligne suivante :
    # df_long['rating'] = df_long['rating'].round()

    # 5. Sauvegarde
    output_path = os.path.join(PROCESSED_DIR, 'jester_processed.csv')
    df_long.to_csv(output_path, index=False)
    print(f"Jester sauvegardé et normalisé : {df_long.shape[0]} notes. -> {output_path}")

def process_netflix():
    print("\n--- Traitement de Netflix (Sous-ensemble) ---")
    # L'article dit : "Random subset of 20,000 users on 2700 movies"
    # ATTENTION : Lire les 4 fichiers combinés prend énormément de RAM.
    # Astuce : On va lire seulement combined_data_1.txt qui contient ~4500 films.
    # C'est suffisant pour extraire un sous-ensemble de 2700 films.
    
    input_file = os.path.join(RAW_DIR, 'netflix', 'combined_data_1.txt')
    
    data = []
    current_movie_id = None
    
    print(f"Lecture de {input_file} (Mode stream)...")
    
    # On va lire ligne par ligne pour ne pas exploser la mémoire
    # On s'arrête si on a trop de données pour ne pas perdre de temps
    max_lines = 5000000 # Lire les 5 premiers millions de lignes pour créer l'échantillon
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                # Format: CustomerID,Rating,Date
                parts = line.split(',')
                user_id = int(parts[0])
                rating = int(parts[1])
                data.append([user_id, current_movie_id, rating])
            
            if i > max_lines:
                break
    
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    
    print(f"Données brutes chargées : {len(df)} lignes.")
    
    # 1. Sélectionner 2700 films aléatoires
    all_movies = df['item_id'].unique()
    if len(all_movies) > 2700:
        selected_movies = np.random.choice(all_movies, 2700, replace=False)
        df = df[df['item_id'].isin(selected_movies)]
    
    # 2. Sélectionner 20 000 utilisateurs aléatoires parmi ceux qui restent
    all_users = df['user_id'].unique()
    if len(all_users) > 20000:
        selected_users = np.random.choice(all_users, 20000, replace=False)
        df = df[df['user_id'].isin(selected_users)]
        
    output_path = os.path.join(PROCESSED_DIR, 'netflix_subset_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"Netflix Subset sauvegardé : {df.shape[0]} notes (Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}). -> {output_path}")

if __name__ == "__main__":
    process_movielens()
    try:
        process_jester()
    except Exception as e:
        print(f"Erreur Jester: {e} (Vérifiez que le fichier xls est bien là)")
        
    try:
        process_netflix()
    except Exception as e:
        print(f"Erreur Netflix: {e}")