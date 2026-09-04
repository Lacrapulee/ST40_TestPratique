import torch
import torch.nn as nn

# Imaginons 1 image découpée en 4 patchs (une grille 2x2 : haut-gauche, haut-droite, etc.)
# Chaque patch a déjà été aplati et passé dans un petit nn.Linear pour faire une taille de 16
# Notre lot (batch) contient 3 images.
# Dimension : (Batch=3, Nombre_de_patches=4, Taille_du_vecteur=16)
patches_entree = torch.rand(3, 4, 16) 

# On crée notre couche d'Attention
# embed_dim=16 : la taille de nos vecteurs de patch
# num_heads=1 : on utilise 1 seule "tête" d'attention pour rester simple
# batch_first=True : dit à PyTorch que notre dimension "3" (le batch) est en première position
couche_attention = nn.MultiheadAttention(embed_dim=16, num_heads=1, batch_first=True)

# On fait passer nos patches dans l'Attention.
# Pour du "Self-Attention", la Query, la Key et la Value sont exactement la même chose : nos patches !
# La fonction renvoie deux choses : les patches enrichis, et les "poids" (qui a regardé qui)
patches_enrichis, poids_qui_regarde_qui = couche_attention(
    query=patches_entree, 
    key=patches_entree, 
    value=patches_entree
)

print("Taille des patches AVANT l'attention :", patches_entree.shape)
print("Taille des patches APRÈS l'attention :", patches_enrichis.shape)
# Résultat : torch.Size([3, 4, 16]). La dimension ne bouge pas !