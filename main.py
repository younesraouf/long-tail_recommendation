STDIN
from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer

def main():
    print("=== Projet MORS: Long Tail Recommendation ===")
    
    # 1. Chargement
    # loader = DataLoader('data/raw/ml-100k')
    
    # 2. CF Model
    # cf = ItemBasedCF(...)
    
    # 3. Optimisation
    # optimizer = MORSOptimizer(...)
    
    print("Projet initialisé. Remplissez les modules dans src/ pour commencer.")

if __name__ == "__main__":
    main()
