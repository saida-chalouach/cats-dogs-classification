import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
IMAGE_SIZE = 150
ZIP_PATH = 'cat-and-dog.zip'

def extract_data():
    print("Extraction des données...")
    if os.path.exists(DATA_DIR):
        print("Le dossier 'data' existe déjà, extraction ignorée.")
        return
    
    os.makedirs(DATA_DIR)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction terminée!")

def find_training_dir():
    """Trouve le bon chemin vers training_set avec cats et dogs"""
    # Chercher récursivement le dossier qui contient cats et dogs
    for root, dirs, files in os.walk(DATA_DIR):
        if 'cats' in dirs and 'dogs' in dirs:
            print(f"Trouvé cats et dogs dans: {root}")
            return root
    
    return None

def load_images():
    print("Chargement des images...")
    images = []
    labels = []
    
    training_dir = find_training_dir()
    
    if training_dir is None:
        print("ERREUR: Impossible de trouver le dossier training_set!")
        print("Contenu du dossier 'data':")
        for item in os.listdir(DATA_DIR):
            print(f"  - {item}")
        return np.array([]), np.array([])
    
    print(f"Dossier training trouvé: {training_dir}")
    
    # Charger les chats
    cats_dir = os.path.join(training_dir, 'cats')
    if not os.path.exists(cats_dir):
        print(f"ERREUR: Le dossier {cats_dir} n'existe pas!")
        return np.array([]), np.array([])
    
    print(f"Chargement des chats...")
    count = 0
    for filename in os.listdir(cats_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(cats_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(0)
                count += 1
                if count % 500 == 0:
                    print(f"  {count} chats chargés...")
                if count >= 2500:
                    break
            except Exception as e:
                print(f"  Erreur avec {filename}: {e}")
    
    print(f"Total chats: {count}")
    
    # Charger les chiens
    dogs_dir = os.path.join(training_dir, 'dogs')
    if not os.path.exists(dogs_dir):
        print(f"ERREUR: Le dossier {dogs_dir} n'existe pas!")
        return np.array(images), np.array(labels)
    
    print(f"Chargement des chiens...")
    count = 0
    for filename in os.listdir(dogs_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dogs_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1)
                count += 1
                if count % 500 == 0:
                    print(f"  {count} chiens chargés...")
                if count >= 2500:
                    break
            except Exception as e:
                print(f"  Erreur avec {filename}: {e}")
    
    print(f"Total chiens: {count}")
    print(f"TOTAL: {len(images)} images chargées!")
    return np.array(images), np.array(labels)

def prepare_data():
    print("Préparation des données...")
    X, y = load_images()
    
    if len(X) == 0:
        print("ERREUR : Aucune image chargée!")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)
    
    print("\n✅ Données sauvegardées avec succès!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

if __name__ == '__main__':
    extract_data()
    prepare_data()