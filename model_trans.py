import torch.nn as nn

# Modèle de transcription vocale basé sur CNN + BiLSTM + couche linéaire
class SpeechToTextModel(nn.Module):
    def __init__(self, vocab_size):
        """
        vocab_size : taille du vocabulaire de sortie (nombre total de tokens possibles).
        """
        super().__init__()
        
        # Partie convolutionnelle (CNN) :
        # Permet d'extraire des caractéristiques locales temporelles et fréquentielles à partir des spectrogrammes Mel.
        self.cnn = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),  # Entrée : 80 canaux (Mel features), sortie : 128
            nn.ReLU(),  # Fonction d'activation non linéaire
            nn.Conv1d(128, 128, kernel_size=3, padding=1),  # Deuxième couche conv avec mêmes dimensions
            nn.ReLU()
        )
        
        # Partie RNN (BiLSTM) :
        # Utilise un LSTM bidirectionnel à deux couches pour capturer les dépendances temporelles dans les deux directions.
        # Entrée : 128 features, sortie : 512 (256*2 pour bidirectionnel)
        self.rnn = nn.LSTM(128, 256, num_layers=2, bidirectional=True, batch_first=True)
        
        # Couche linéaire finale :
        # Projette la sortie du BiLSTM (512) vers la taille du vocabulaire
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        """
        x : tenseur d’entrée de forme (batch_size, 80, time_steps)
        Retourne : logits de forme (batch_size, time_steps, vocab_size)
        """
        # Passage par CNN 
        x = self.cnn(x)  # Résultat : (batch_size, 128, time_steps)
        
        #  Transposition pour l’entrée dans le LSTM : (batch, time_steps, features)
        x = x.transpose(1, 2)
        
        # Passage dans le BiLSTM
        x, _ = self.rnn(x)  # Résultat : (batch_size, time_steps, 512)
        
        # Passage par la couche finale pour obtenir les scores de chaque token
        x = self.fc(x)  # Résultat : (batch_size, time_steps, vocab_size)
        
        return x