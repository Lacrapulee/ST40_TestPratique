import torch
import torch.nn as nn

# 1. Le petit réseau classique (Linear) qu'on a vu au début : ils l'appellent MLP
class Mlp(nn.Module):
    def __init__(self, dim_entree, dim_cachee):
        super().__init__()
        # On retrouve nos nn.Linear ! 
        # (Généralement dim_cachee est 4 fois plus grande que dim_entree)
        self.fc1 = nn.Linear(dim_entree, dim_cachee)
        self.act = nn.GELU() # C'est comme ReLU, une fonction d'activation
        self.fc2 = nn.Linear(dim_cachee, dim_entree)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 2. La fameuse brique de base : Le Block
class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        # Norm1 et Norm2 servent juste à stabiliser les calculs (LayerNorm)
        self.norm1 = nn.LayerNorm(dim)
        # Voici la couche d'Attention !
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        # Voici le réseau de neurones classique
        self.mlp = Mlp(dim_entree=dim, dim_cachee=dim * 4)

    def forward(self, x):
        # ÉTAPE A : Les patches communiquent entre eux (Attention)
        x_norm = self.norm1(x)
        attention_sortie, _ = self.attn(x_norm, x_norm, x_norm) # Query, Key, Value
        x = x + attention_sortie  # On ajoute le contexte au patch original
        
        # ÉTAPE B : Chaque patch est compressé/transformé individuellement (Linear)
        x = x + self.mlp(self.norm2(x))
        
        return x


class Encodeur_IJEPA(nn.Module):
    def __init__(self, dim=768, num_heads=12, profondeur=12):
        super().__init__()
        # On empile 12 Blocks à la suite !
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads) for _ in range(profondeur)
        ])
        
    def forward(self, x):
        # La donnée traverse les 12 blocs, un par un
        for block in self.blocks:
            x = block(x)
        return x


# 1. Création d'un encodeur et simulation de sauvegarde (les poids sont créés aléatoirement)
mon_encodeur_entraine = Encodeur_IJEPA(dim=256, num_heads=8, profondeur=6)
# On sauvegarde son "cerveau" (ses poids) dans un fichier .pth (format standard PyTorch)
torch.save(mon_encodeur_entraine.state_dict(), "../data/poids_ijepa.pth")

# --- DÉBUT DU TEST RÉEL ---

# 2. On instancie un nouvel encodeur, qui naît complètement "ignorant"
nouvel_encodeur = Encodeur_IJEPA(dim=256, num_heads=8, profondeur=6)

# 3. LE CHARGEMENT : On injecte les poids pré-entraînés dans ce modèle vide
poids_sauvegardes = torch.load("../data/poids_ijepa.pth")
nouvel_encodeur.load_state_dict(poids_sauvegardes)
print("Poids chargés avec succès !")

# 4. Mode Évaluation (Crucial !)
# Bloque certains comportements spécifiques à l'entraînement (comme le Dropout)
nouvel_encodeur.eval() 

# 5. Création d'une fausse image sous forme de patches (Batch=1, 196 patches, dimension=256)
# Dans la réalité, on utiliserait une couche spéciale pour découper l'image RGB en ces patches
image_test_patches = torch.rand(1, 196, 256)

# 6. L'inférence (le test)
# torch.no_grad() dit à PyTorch : "Ne calcule pas les gradients, on ne fait que tester, pas entraîner"
# Ça économise énormément de mémoire et de temps.
with torch.no_grad():
    representations_finales = nouvel_encodeur(image_test_patches)

print("Taille de l'image en entrée :", image_test_patches.shape)
print("Taille des représentations en sortie :", representations_finales.shape)
# Résultat : torch.Size([1, 196, 256]). L'encodeur a enrichi chaque patch sans changer la taille !