import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None, sample_rate=16000, n_mels=64, max_frames=32):
        # Initialisation du dataset avec les paramètres :
        # data_dir : chemin vers le dossier contenant les sous-dossiers des classes audio
        # classes : liste des noms des classes (ex: ['yes', 'no', ...])
        # transform : transformations optionnelles à appliquer sur les spectrogrammes
        # sample_rate : fréquence d'échantillonnage à utiliser pour charger les audios
        # n_mels : nombre de bandes Mel pour les spectrogrammes
        # max_frames : nombre fixe de frames temporelles (colonnes) pour chaque spectrogramme

        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_frames = max_frames  # Nombre fixe de frames pour chaque spectrogramme
        self.files = []   # Liste qui stockera les chemins vers les fichiers audio
        self.labels = []  # Liste qui stockera les labels correspondants (indices des classes)

        # Parcours des dossiers de chaque classe pour récupérer les fichiers .wav
        for label_idx, label in enumerate(classes):
            label_path = os.path.join(data_dir, label)
            if not os.path.exists(label_path):
                # Affiche un avertissement si un dossier de classe est manquant
                print(f"Attention : dossier manquant pour la classe '{label}'")
                continue
            for file in os.listdir(label_path):
                if file.endswith(".wav"):  # On prend uniquement les fichiers .wav
                    self.files.append(os.path.join(label_path, file))
                    self.labels.append(label_idx)  # L'index de la classe est le label

        # Si aucun fichier audio n'a été trouvé, on lève une erreur
        if len(self.files) == 0:
            raise RuntimeError("Aucun fichier audio trouvé dans les dossiers spécifiés.")

    def __len__(self):
        # Retourne le nombre total d'exemples dans le dataset
        return len(self.files)

    def __getitem__(self, idx):
        # Chargement d'un exemple et extraction de ses features à l'index idx

        file_path = self.files[idx]  # Chemin du fichier audio
        label = self.labels[idx]     # Label associé

        try:
            # Chargement du fichier audio avec librosa (re-échantillonnage au besoin)
            y, sr = librosa.load(file_path, sr=self.sample_rate)
        except Exception as e:
            # En cas d'erreur de chargement, affiche un message et retourne un spectrogramme vide + label invalide (-1)
            print(f"Erreur chargement fichier audio '{file_path}': {e}")
            empty_spec = torch.zeros(1, self.n_mels, self.max_frames)
            return empty_spec, -1

        try:
            # Calcul du spectrogramme Mel du signal audio
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            # Conversion en décibels (logarithmique) pour une meilleure représentation visuelle et apprentissage
            S_dB = librosa.power_to_db(S, ref=np.max)
        except Exception as e:
            # En cas d'erreur lors de l'extraction du spectrogramme, même gestion que ci-dessus
            print(f"Erreur extraction spectrogramme pour '{file_path}': {e}")
            empty_spec = torch.zeros(1, self.n_mels, self.max_frames)
            return empty_spec, -1

        # Padding (ajout de colonnes de zeros) ou tronquage du spectrogramme pour avoir une taille fixe en frames
        if S_dB.shape[1] < self.max_frames:
            pad_width = self.max_frames - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0,0), (0,pad_width)), mode='constant')
        else:
            S_dB = S_dB[:, :self.max_frames]

        # Conversion en tenseur PyTorch et ajout d'une dimension canal 
        S_dB = torch.tensor(S_dB).unsqueeze(0).float()

        # Application éventuelle d'une transformation complémentaire passée en paramètre
        if self.transform:
            S_dB = self.transform(S_dB)

        # Retourne le spectrogramme (tensor) et le label associé
        return S_dB, label