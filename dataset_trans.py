import sys
import os
import torch
from torch.utils.data import Dataset

# ✅ Ajoute le chemin racine du projet pour les imports éventuels
sys.path.append(r"C:\Users\Boumahdi\Documents\hayat\S2\ml\projet final")

class SpeechToTextDataset(Dataset):
    """
    Dataset personnalisé pour la transcription vocale (speech-to-text).

    Ce dataset prend en entrée :
    - un dataset Hugging Face contenant des paires audio/texte
    - un tokenizer pour transformer les transcriptions en tokens

    Retourne pour chaque élément :
    - input_feat : tenseur contenant les caractéristiques audio (ex. : spectrogrammes Mel)
    - labels : séquence de tokens représentant la transcription
    """

    def __init__(self, hf_dataset, tokenizer):
        """
        Initialise le dataset.

        Args:
            hf_dataset: Dataset Hugging Face avec les champs "input_features" et "sentence"
            tokenizer: Tokenizer compatible (ex : WhisperTokenizer, Wav2Vec2Tokenizer)
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Retourne la taille du dataset (nombre d'exemples).
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Récupère un exemple du dataset à l'index `idx`.

        Args:
            idx: Index de l'exemple

        Returns:
            Tuple (input_feat, labels)
            - input_feat: torch.FloatTensor des features audio
            - labels: torch.LongTensor des IDs de tokens du texte
        """
        item = self.dataset[idx]
        input_feat = torch.tensor(item["input_features"], dtype=torch.float32)
        labels = self.tokenizer(item["sentence"], return_tensors="pt", padding="longest").input_ids[0]
        return input_feat, labels