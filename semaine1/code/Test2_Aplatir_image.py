import torch
import torch.nn as nn

# En PyTorch, une image est toujours définie par : (Batch, Canaux_RGB, Hauteur, Largeur)
# Créons un lot de 3 images couleur de taille 10x10 pixels
images_batch = torch.rand(3, 3, 10, 10)
print("Forme de départ :", images_batch.shape)

# On utilise l'outil Flatten pour aplatir tout ce qui se trouve après la dimension Batch
aplatisseur = nn.Flatten(start_dim=1)
images_aplaties = aplatisseur(images_batch)

# Résultat : 3 images, transformées chacune en 300 valeurs (3 * 10 * 10)
print("Forme après aplatissement :", images_aplaties.shape)

# Maintenant, on peut envoyer ça dans notre Encodeur de tout à l'heure !
# Il faut juste que l'encodeur ait dimension_entree=300
encodeur_image = nn.Sequential(
    nn.Linear(300, 128),
    nn.ReLU(),
    nn.Linear(128, 16)
)

embeddings = encodeur_image(images_aplaties)
print("Taille de sortie :", embeddings.shape) 
# Sortie : torch.Size([3, 16]) -> On a bien 3 vecteurs compressés de taille 16