# download_dataset_trans.py

from datasets import load_dataset, Audio
import os

# Chemin vers le dossier Common Voice extrait 
cv_path = r"C:\Users\Boumahdi\Documents\hayat\S2\ml\projet final\data\cv-corpus-7.0-singleword\fr"

# Définir les fichiers tsv pour train, validation, test
data_files = {
    "train": os.path.join(cv_path, "train.tsv"),
    "validation": os.path.join(cv_path, "validation.tsv"),
    "test": os.path.join(cv_path, "test.tsv")
}

# Charger dataset à partir des tsv
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# Ajouter la colonne 'audio' pointant vers les fichiers audio (clips)
def add_audio_path(batch):
    # 'path' est dans le tsv, ex: 'common_voice_fr_27436515.mp3'
    batch["audio"] = {"path": os.path.join(cv_path, "clips", batch["path"])}
    return batch

dataset = dataset.map(add_audio_path)

# Convertir la colonne audio en type Audio avec sampling_rate 16000
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Dataset chargé ")
print(dataset["train"][0])
