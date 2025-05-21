import re
import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from model import SpeechToTextModel
from dataset import SpeechToTextDataset
from transformers import AutoTokenizer
from download_dataset import dataset  
from audio_feature_extraction import extract_features

# Fonction pour nettoyer les textes (transcriptions) en retirant caractères spéciaux inutiles
def remove_special_characters(batch):
    # Par exemple, garder uniquement lettres, chiffres, espaces et apostrophes
    batch["sentence"] = re.sub(r"[^a-zA-Z0-9\s']", "", batch["sentence"]).lower()
    return batch

# Nettoyage des textes dans le dataset
dataset = dataset.map(remove_special_characters)

# Extraction des features audio (spectrogrammes, etc.)
dataset = dataset.map(extract_features)

# Création du tokenizer (vocabulaire) et du DataLoader
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

train_dataset = SpeechToTextDataset(dataset["train"], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

# Initialisation du modèle, fonction de perte et optimiseur
model = SpeechToTextModel(vocab_size=tokenizer.vocab_size)
loss_fn = CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

# Boucle d'entraînement
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        inputs, targets = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        logits = model(inputs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        input_lengths = torch.full((len(inputs),), logits.size(1), dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets])
        targets = torch.cat(targets)

        loss = loss_fn(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader)}")

torch.save(model.state_dict(), "model.pth")