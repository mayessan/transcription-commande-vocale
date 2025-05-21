import torch
import torchaudio
from transformers import AutoTokenizer
from audio_feature_extraction_trans import extract_features
from model_trans import SpeechToTextModel
from pydub import AudioSegment
import os

# Convertit un fichier MP3 en WAV (nécessaire car torchaudio charge plus facilement les fichiers WAV)
def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Effectue la prédiction sur un fichier audio donné (WAV ou converti depuis MP3)
def predict_audio(model, tokenizer, audio_path):
    # Chargement du fichier audio avec torchaudio
    waveform, sr = torchaudio.load(audio_path)
    print("Shape du waveform avant mono:", waveform.shape)  

    # Conversion en mono si le fichier est en stéréo (2 canaux)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    print("Shape du waveform après mono:", waveform.shape)  

    # Retirer la dimension du canal pour être compatible avec l'extraction de features
    waveform = waveform.squeeze(0)

    # Préparer l'exemple sous forme de dictionnaire pour la fonction d'extraction
    example = {"audio": {"array": waveform.numpy(), "sampling_rate": sr}}
    example = extract_features(example)  # génère les spectrogrammes Mel
    features = example["input_features"]
    print("Shape des features:", features.shape)  # debug

    # Mise en forme pour Conv1D : (batch=1, channels=n_mels, time_steps)
    inputs = features.permute(1, 0).unsqueeze(0)
    print("Shape des inputs:", inputs.shape)  # debug

    # Prédiction
    model.eval()
    with torch.no_grad():
        logits = model(inputs)

    # Appliquer softmax log pour la classification
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    predicted_ids = torch.argmax(log_probs, dim=-1)[0].tolist()

    # Décodage CTC : supprimer les doublons et les blancs (pad_token_id)
    blank_id = tokenizer.pad_token_id
    decoded = []
    previous = None
    for p in predicted_ids:
        if p != previous and p != blank_id:
            decoded.append(p)
        previous = p

    # Décodage final en texte
    transcription = tokenizer.decode(decoded, skip_special_tokens=True)
    return transcription

# Point d'entrée principal du script
if __name__ == "__main__":
    import argparse

    # Parser des arguments pour spécifier un fichier audio
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_audio", type=str, required=True)
    args = parser.parse_args()

    mp3_path = args.predict_audio
    wav_path = "temp_audio.wav"

    # Conversion si nécessaire
    if mp3_path.lower().endswith(".mp3"):
        mp3_to_wav(mp3_path, wav_path)
        audio_path = wav_path
    else:
        audio_path = mp3_path

    # Chargement du tokenizer et du modèle
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = SpeechToTextModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))

    # Transcription finale
    result = predict_audio(model, tokenizer, audio_path)
    print("Transcription :", result)

    #  Nettoyage du fichier temporaire
    if os.path.exists(wav_path):
        os.remove(wav_path)