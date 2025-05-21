import torch
from torch.utils.data import DataLoader
from model_trans import SpeechToTextModel              # Modèle de transcription audio vers texte
from dataset_trans import SpeechToTextDataset          # Dataset personnalisé compatible PyTorch
from transformers import AutoTokenizer                 # Tokenizer pour encoder/decoder le texte
from download_dataset_trans import dataset             # Dataset Hugging Face Common Voice (ou autre)
from audio_feature_extraction_trans import extract_features  # Fonction d'extraction de features audio
import torchaudio
import argparse

# Fonction d'évaluation du modèle sur le jeu de validation
def evaluate(model, dataloader, loss_fn, tokenizer):
    model.eval()                   # Met le modèle en mode évaluation (désactive dropout, etc.)
    total_loss = 0                 # Accumulateur de perte
    with torch.no_grad():         # Désactive la rétropropagation pour accélérer
        for batch in dataloader:
            inputs, targets = zip(*batch)   # Séparer les entrées et les cibles
            inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)  # Padding pour batch
            logits = model(inputs)         # Prédiction brute du modèle
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)         # Softmax log pour CTC
            input_lengths = torch.full((len(inputs),), logits.size(1), dtype=torch.long)  # Longueurs entrées
            target_lengths = torch.tensor([len(t) for t in targets])            # Longueurs cibles
            targets = torch.cat(targets)                                        # Concaténation des cibles
            # Calcul de la perte CTC
            loss = loss_fn(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(dataloader)}")   # Affiche la perte moyenne

# Fonction pour faire une prédiction sur un fichier audio
def predict(model, tokenizer, audio_path):
    waveform, sr = torchaudio.load(audio_path)   # Charge le fichier audio
    example = {
        "audio": {"array": waveform.squeeze().numpy(), "sampling_rate": sr}
    }
    example = extract_features(example)            # Extraction de spectrogrammes Mel
    inputs = example["input_features"].unsqueeze(0)  # Ajout d'une dimension batch
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Log-softmax pour CTC
    predicted_ids = torch.argmax(log_probs, dim=-1)              # Indices prédits (vocabulaire)
    transcription = tokenizer.decode(predicted_ids[0].tolist())  # Décodage vers texte
    return transcription

# Fonction principale : évalue le modèle et prédit si un fichier est fourni
def main():
    parser = argparse.ArgumentParser(description="Évaluation et prédiction d'un modèle Speech-to-Text")
    parser.add_argument("--predict_audio", type=str, help="Chemin vers un fichier audio pour faire une prédiction")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")  # Tokenizer Wav2Vec2
    model = SpeechToTextModel(vocab_size=tokenizer.vocab_size)                # Modèle défini avec le vocab
    model.load_state_dict(torch.load("model.pth"))                            # Chargement des poids
    loss_fn = torch.nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)  #  Perte CTC

    # Chargement et prétraitement du dataset de validation
    dataset_local = dataset.map(extract_features)  #  Application de l'extraction sur tout le jeu
    val_dataset = SpeechToTextDataset(dataset_local["validation"], tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)

    # Évaluation du modèle
    evaluate(model, val_loader, loss_fn, tokenizer)

    # Si fichier audio fourni, effectuer une prédiction
    if args.predict_audio:
        transcription = predict(model, tokenizer, args.predict_audio)
        print(f"Transcription audio '{args.predict_audio}': {transcription}")

# Point d’entrée principal du script
if __name__ == "__main__":
    main()