import torch
import torch.nn as nn

# 1. On crée notre classe qui hérite du fameux "père" nn.Module
class Encodeur(nn.Module):
    def __init__(self, dimension_entree, dimension_sortie):
        # super().__init__() est obligatoire : il initialise la classe mère nn.Module
        super().__init__() 
        
        # On définit nos couches (les "poids" du modèle)
        self.couches = nn.Sequential(
            nn.Linear(dimension_entree, 128),
            nn.ReLU(),
            nn.Linear(128, dimension_sortie)
        )

    # 2. On définit le chemin de la donnée
    def forward(self, x):
        representation = self.couches(x)
        return representation

# --- MISE EN PRATIQUE ---

# On détermine si un GPU est dispo, sinon on reste sur CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"On utilise le processeur : {device.upper()}")

# On instancie notre encodeur (ex: il prend 64 valeurs et en ressort 16)
mon_encodeur = Encodeur(dimension_entree=64, dimension_sortie=16)

# On utilise la méthode héritée de nn.Module pour l'envoyer sur le GPU
mon_encodeur = mon_encodeur.to(device)

# On crée une fausse donnée (batch de 3 exemples, chacun de taille 64)
# On doit AUSSI envoyer la donnée sur le même device que le modèle
fausses_donnees = torch.rand(3, 64).to(device)

# On fait passer la donnée dans le modèle (PyTorch appelle automatiquement la fonction forward)
embeddings = mon_encodeur(fausses_donnees)

print("Taille de la sortie de l'encodeur :", embeddings.shape)
# Résultat attendu : torch.Size([3, 16])