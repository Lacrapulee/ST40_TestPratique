import torch
import torch.nn as nn

class PredictorIJEPAAvecMasque(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)

    # Désormais, on prend en compte les positions des patches cachés
    def forward(self, representation_contexte, positions_masquees):
        # Le prédicteur combine le contexte et les positions à deviner
        x = self.linear1(representation_contexte)
        
        # Si on fournit des positions masquées, on peut les additionner ou les concaténer
        if positions_masquees is not None:
            x = x + positions_masquees
            
        x = self.gelu(x)
        prediction = self.linear2(x)
        return prediction

# Test rapide
predictor = PredictorIJEPAAvecMasque(dim=256)
contexte_encode = torch.rand(2, 50, 256)      # 50 patches visibles encodés
positions_caches = torch.rand(2, 50, 256)     # Les positions géographiques des patches cachés

prediction_cible = predictor(contexte_encode, positions_caches)
print("Forme de la prédiction avec masques :", prediction_cible.shape)
# Résultat : torch.Size([2, 50, 256])