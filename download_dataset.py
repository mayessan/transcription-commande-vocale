import os
import urllib.request
import tarfile

# URL du jeu de données Speech Commands (version 0.02) hébergé par TensorFlow
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Dossier où les données seront extraites après téléchargement
data_dir = "./speech_commands"

#  Création du dossier cible s'il n'existe pas déjà
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Nom du fichier tar.gz téléchargé
tar_path = "speech_commands_v0.02.tar.gz"

# Étape 1 : Télécharger le fichier .tar.gz depuis l'URL
print("Téléchargement en cours...")
urllib.request.urlretrieve(url, tar_path)
print("Téléchargement terminé.")

# Étape 2 : Extraire le contenu de l'archive dans le dossier spécifié
print("Extraction en cours...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=data_dir)
print("Extraction terminée.")