import torch
import torch.nn as nn
import copy

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class IJEPAEncoder(nn.Module):
    def __init__(self, dim=256, num_heads=8, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class IJEPAPredictor(nn.Module):
    def __init__(self, dim=256, num_heads=8, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.predict_head = nn.Linear(dim, dim)

    def forward(self, context, mask_tokens):
        # 1. On combine contexte (75) et tokens masqués (25) -> Total 100
        x = torch.cat([context, mask_tokens], dim=1)
        
        # 2. Passage dans les couches du prédicteur
        for block in self.blocks:
            x = block(x)
            
        # 3. On extrait uniquement les prédictions des tokens masqués (les 25 derniers)
        predictions = self.predict_head(x[:, -mask_tokens.shape[1]:, :])
        return predictions

# --- SIMULATION D'UNE ÉTAPE D'ENTRAÎNEMENT ---

dim = 256
batch_size = 2
num_context = 75
num_target = 25

# Instanciation des modèles
encoder_context = IJEPAEncoder(dim=dim)
encoder_target = copy.deepcopy(encoder_context) # L'encodeur cible copie l'encodeur de contexte

# Désactivation des gradients pour l'encodeur cible
for p in encoder_target.parameters():
    p.requires_grad = False

predictor = IJEPAPredictor(dim=dim)
optimizer = torch.optim.AdamW(list(encoder_context.parameters()) + list(predictor.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# Fausses données : 75 patches visibles et 100 patches au total (75 contexte + 25 cibles)
context_inputs = torch.rand(batch_size, num_context, dim)
target_inputs_full = torch.rand(batch_size, num_context + num_target, dim)
mask_tokens_with_pos = torch.rand(batch_size, num_target, dim) # Contient les positions cachées

# A. L'encodeur cible traite toute l'image pour fournir les vraies références
with torch.no_grad():
    target_representation = encoder_target(target_inputs_full)
    true_targets = target_representation[:, -num_target:, :] # On isole les 25 cibles cachées

# B. L'encodeur de contexte traite les parties visibles
context_representation = encoder_context(context_inputs)

# C. Le prédicteur combine contexte et masques pour deviner les cibles
predicted_targets = predictor(context_representation, mask_tokens_with_pos)

# D. Calcul de l'erreur et rétropropagation
loss = criterion(predicted_targets, true_targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss d'entraînement calculée : {loss.item():.4f}")