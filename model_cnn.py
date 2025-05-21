import torch
import torch.nn as nn
import torch.nn.functional as F

# Définition d’un réseau de neurones convolutif simple (CNN) pour la classification audio
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=12, input_shape=(1, 64, 32)):
        """
        num_classes : nombre de classes cibles à prédire.
        input_shape : forme des entrées audio converties en spectrogrammes (C, H, W).
        """
        super(SimpleCNN, self).__init__()
        
        # Première couche convolutionnelle : 1 canal -> 16 cartes de caractéristiques
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        
        # Couche de pooling (réduction de dimensions)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Deuxième couche convolutionnelle : 16 -> 32 canaux
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # Calcul automatique de la taille de la sortie  après les convolutions et pooling
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # simulation d'une entrée (batch=1)
            x = self.pool(F.relu(self.conv1(x)))  # passage par conv1 + relu + pool
            x = self.pool(F.relu(self.conv2(x)))  # passage par conv2 + relu + pool
            self.flattened_size = x.numel()       # nombre total de neurones à l’entrée du FC

        # Première couche entièrement connectée
        self.fc1 = nn.Linear(self.flattened_size, 128)

        # Couche de sortie : 128 → num_classes (classification)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Propagation avant du signal à travers le modèle :
        conv1 → ReLU → pool → conv2 → ReLU → pool → flatten → fc1 → ReLU → fc2
        """
        x = self.pool(F.relu(self.conv1(x)))  # Bloc conv1
        x = self.pool(F.relu(self.conv2(x)))  # Bloc conv2

        x = torch.flatten(x, start_dim=1)     # Aplatissement des features (hors batch)

        x = F.relu(self.fc1(x))               # FC1 + ReLU
        x = self.fc2(x)                       # FC2 : sortie finale (logits)
        return x