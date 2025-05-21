import librosa
import numpy as np
import matplotlib.pyplot as plt

# Chemin vers un fichier audio .wav de la commande vocale "yes"
filename = './speech_commands/yes/0a7c2a8d_nohash_0.wav'

# Chargement de l'audio avec librosa
# sr=16000 : on force la fréquence d'échantillonnage à 16 kHz
y, sr = librosa.load(filename, sr=16000)

# Affichage de la forme d'onde (waveform) du signal audio
plt.figure(figsize=(10, 2))  # Taille de la figure (largeur x hauteur en pouces)
plt.plot(y)                  # Trace la forme d'onde (amplitude en fonction du temps)
plt.title('Waveform')        # Titre du graphique
plt.show()                   # Affiche la figure

# Calcul du spectrogramme Mel du signal audio
# n_mels=64 : nombre de bandes Mel (fréquences perceptuelles)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)

# Conversion en décibels pour une meilleure visualisation
# ref=np.max : référence au maximum pour la normalisation
S_dB = librosa.power_to_db(S, ref=np.max)

# Affichage du spectrogramme Mel
plt.figure(figsize=(10, 4))          # Taille de la figure
plt.title('Mel spectrogram')          # Titre du graphique

# Affiche la représentation du spectrogramme Mel dans le domaine temps-fréquence
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB')    # Barre de couleur avec échelle en décibels
plt.show()                         # Affiche la figure