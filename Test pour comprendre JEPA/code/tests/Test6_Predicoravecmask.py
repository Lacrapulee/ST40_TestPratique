import torch
import torch.nn as nn

class PredictorIJEPAPropre(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, representation_contexte, tokens_masques_avec_positions):
        # representation_contexte : (Batch=2, 75 patches, 256)
        # tokens_masques_avec_positions : (Batch=2, 25 patches cachés, 256)
        
        # 1. On concatène le contexte et les masques sur la dimension des patches (dim=1)
        # Résultat : (2, 75 + 25 = 100, 256)
        entree_combinee = torch.cat([representation_contexte, tokens_masques_avec_positions], dim=1)
        
        # 2. On fait passer le tout dans le réseau (qui mélange l'information)
        traitement = self.linear(entree_combinee)
        
        # 3. On découpe pour ne garder QUE la prédiction des 25 zones cachées (les 25 derniers)
        prediction_finale = traitement[:, -25:, :]
        
        return prediction_finale

# Test des dimensions
predictor = PredictorIJEPAPropre(dim=256)
contexte = torch.rand(2, 75, 256)       # 75 visibles
masques = torch.rand(2, 25, 256)        # 25 positions cachées

sortie = predictor(contexte, masques)
print("Forme de la sortie du prédicteur :", sortie.shape)
# Résultat : torch.Size([2, 25, 256]) -> On retombe exactement sur nos 25 cibles !