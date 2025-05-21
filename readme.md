Projet Deep Learning Audio : Détection de Commandes Vocales & Transcription Vocale
**Description
Ce projet implémente deux pipelines audio en Deep Learning :

*Détection de commandes vocales
Utilise le dataset Google Speech Commands pour reconnaître des commandes vocales simples (ex : "yes", "no", "stop", etc.) via un modèle CNN.

*Transcription vocale automatique (speech-to-text)
Utilise le dataset Common Voice (français) pour entraîner un modèle CNN + BiLSTM avec CTC Loss, capable de transcrire l’audio en texte.

Ces deux approches couvrent la préparation des données, extraction des features, définition des modèles, entraînement, évaluation et interfaces simples.

**Prérequis
Python 3.8 ou plus

CUDA (optionnel pour accélération GPU)

Librairies Python (installées dans un environnement virtuel) :

pip install torch torchaudio librosa matplotlib numpy pandas scikit-learn streamlit gradio seaborn datasets transformers jiwer sentencepiece protobuf
**Structure du projet

** pour la transcription
├── download_dataset_trans.py         	  # Téléchargement et extraction des datasets
├── dpreprocess_data.py          	  # Nettoie et normalise les transcriptions en supprimant la ponctuation.
├── audio_feature_extraction_trans.py 	  # Extraction des spectrogrammes Mel
├── dataset_trans.py       	          # Définit un Dataset PyTorch personnalisé pour charger audio et labels.
├── model_trans.py                   	  # Définit le modèle Deep Learning (CNN + BiLSTM) pour la transcription.
├── train_trans.py                    	  # Entraîne le modèle avec la fonction de perte CTC sur les données préparées.
├── run_model.py                	  # Évalue le modèle sur un dataset de validation et réalise des prédictions sur fichier audio.


** pour la commande_vocale
├── download_dataset.py         	  # Téléchargement et extraction du dataset Google Speech Commands.
├── audio_feature_extraction.py 	  # Affiche un exemple de spectrogramme Mel généré à partir d’un fichier audio.
├── dataset.py                  	  # Crée un Dataset PyTorch personnalisé à partir des fichiers audio en spectrogrammes.
├── model_cnn.py                	  # Définit un modèle CNN simple pour la classification des commandes vocales.
├── train.py                     	  # Entraîne le modèle CNN sur les spectrogrammes extraits des commandes vocales.
├── evaluate.py                  	  # Évaluation modèle commande vocale..

app.py					                    #Affiche une interface gradio pour tester à la fois la transcription audio et la détection de commandes vocales.


**Installation et configuration
*Installer Python 3.8+ depuis python.org

*Créer un environnement virtuel et l’activer

python -m venv dl_audio_env
dl_audio_env\Scripts\activate

*Installer les dépendances :


pip install torch torchaudio librosa matplotlib numpy pandas scikit-learn streamlit gradio seaborn datasets transformers jiwer sentencepiece protobuf

**Usage

1. Télécharger les datasets
*Pour Google Speech Commands :

python download_dataset_trans.py --> transcription

*Pour Common Voice (français, 10%) :

python download_dataset.py --> commande_vocale


3. Entraînement

python train.py --> pour la commande vocale
python train_trans.py --> pour la commande vocale

4. Évaluation
*Pour la détection commandes vocales :

python evaluate.py

*Pour la transcription speech-to-text :

python run_model.py 

5.app.py

python app.py --> pour lancer l'application
