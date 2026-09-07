import os
from datasets import load_dataset

print(" Chargement d'un extrait de 10 000 images de Food-101 depuis Hugging Face...")
# On charge 10 000 images depuis le jeu d'entraînement de food101
dataset = load_dataset("ethz/food101", split="train[:10000]")

# Récupération des noms textuels des classes (ex: 'apple_pie', 'pizza'...)
class_names = dataset.features["label"].names

output_dir = "/workspace/data/train"
os.makedirs(output_dir, exist_ok=True)

print(f"sOrganisation et écriture de {len(dataset)} images dans {output_dir}...")

for idx, item in enumerate(dataset):
    image = item["image"]
    label_idx = item["label"]
    class_name = class_names[label_idx]
    
    # Création du sous-dossier de la classe s'il n'existe pas
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Sauvegarde de l'image au format JPG
    image.convert("RGB").save(os.path.join(class_dir, f"img_{idx}.jpg"))
    
    # Petite indication visuelle de progression
    if (idx + 1) % 1000 == 0:
        print(f"-> {idx + 1} images traitées...")

print(" Téléchargement et organisation terminés avec succès !")