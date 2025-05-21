import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Import du dataset personnalisé et du modèle CNN
from dataset import SpeechCommandsDataset
from model_cnn import SimpleCNN

# Fonction pour afficher la matrice de confusion de manière lisible
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.title("Matrice de confusion")
    plt.show()

# Fonction d’évaluation du modèle CNN entraîné
def evaluate():
    # Liste des commandes cibles à détecter
    labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    
    # Chemin vers le dossier contenant les fichiers audio
    data_dir = "./speech_commands"
    
    # Chemin vers le fichier de poids du modèle
    model_path = "simple_cnn_speech_commands.pth"

    # Vérification de l’existence du dossier de données
    if not os.path.exists(data_dir):
        print(f"Erreur : le dossier des données '{data_dir}' n'existe pas.")
        return

    # Vérification de l’existence du fichier de poids
    if not os.path.exists(model_path):
        print(f"Erreur : le fichier de poids du modèle '{model_path}' est introuvable.")
        return

    # Chargement du dataset
    try:
        dataset = SpeechCommandsDataset(data_dir, labels)
    except Exception as e:
        print(f"Erreur lors du chargement du dataset : {e}")
        return

    # Vérification si le dataset est vide
    if len(dataset) == 0:
        print("Erreur : le dataset est vide.")
        return

    # Séparation du dataset en 80% train et 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    # Création du DataLoader pour le jeu de test
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Utilisation du GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chargement du modèle et transfert vers le bon device
    model = SimpleCNN(num_classes=len(labels)).to(device)

    # Chargement des poids du modèle entraîné
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Erreur lors du chargement des poids du modèle : {e}")
        return

    model.eval()  # Passage en mode évaluation

    all_preds = []   # Prédictions du modèle
    all_labels = []  # Labels réels

    # Boucle d’inférence sur le jeu de test
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Rapport de classification (précision, rappel, F1-score)
    print("Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=labels))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("Matrice de confusion :")
    print(cm)

    # Affichage visuel de la matrice
    plot_confusion_matrix(cm, labels)

# Point d’entrée du script
if __name__ == "__main__":
    evaluate()