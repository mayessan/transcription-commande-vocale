import torch
import torchaudio

# Définition de la transformation MelSpectrogram avec torchaudio
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,   # Fréquence d'échantillonnage en Hz
    n_fft=400,           # Taille de la fenêtre FFT
    win_length=None,     # Longueur de la fenêtre, None => égale à n_fft
    hop_length=160,      # Pas entre deux fenêtres successives (overlap)
    f_min=0,             # Fréquence minimale considérée dans le spectrogramme
    f_max=8000,          # Fréquence maximale considérée (ici la moitié de 16kHz)
    n_mels=80            # Nombre de bandes Mel (représentation perceptuelle)
)

def extract_features(example):
    # Récupère le signal audio sous forme de tableau numpy
    waveform = example["audio"]["array"]

    # Conversion en tenseur PyTorch de type float32
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

    # Si le tenseur est mono (1 dimension), on ajoute une dimension batch (channel)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    # Application de la transformation MelSpectrogram
    # Résultat : tenseur (n_mels, time), on transpose en (time, n_mels)
    mel = mel_transform(waveform_tensor).squeeze(0).transpose(0, 1)  # (time, n_mels)

    # Ajout des features extraites dans le dictionnaire de l'exemple
    example["input_features"] = mel

    # Retourne l'exemple modifié avec les features Mel spectrogram
    return example