import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from dataset import SpeechCommandsDataset
from model_cnn import SimpleCNN  
def main():
    try:
        # Liste des commandes cibles
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        data_dir = "./speech_commands"

        # Vérifier si les dossiers de classes existent
        for label in labels:
            if not os.path.exists(os.path.join(data_dir, label)):
                raise FileNotFoundError(f"Dossier '{label}' manquant dans {data_dir}")

        # Charger dataset complet
        dataset = SpeechCommandsDataset(data_dir, labels)

        if len(dataset) == 0:
            raise ValueError("Aucun fichier audio trouvé dans le dataset.")

        # Diviser en entraînement/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # DataLoaders avec num_workers=0 pour éviter blocage sous Windows
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

        print(f"Nombre batches entraînement : {len(train_loader)}")
        print(f"Nombre batches test : {len(test_loader)}")

        # Initialisation du modèle, perte, optimiseur
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Appareil utilisé : {device}")

        model = SimpleCNN(num_classes=len(labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Entraînement
        for epoch in range(5):
            model.train()
            total_loss = 0
            print(f"Début époque {epoch+1}")
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)}")
                inputs, targets = inputs.to(device), targets.to(device)

                # Vérification de la forme d'entrée
                if inputs.ndim != 4:
                    raise RuntimeError(f"Entrée invalide: attendue 4D, reçue {inputs.ndim}D")

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Époque {epoch+1} - Perte moyenne : {total_loss / len(train_loader):.4f}")

        # Sauvegarde du modèle
        torch.save(model.state_dict(), "simple_cnn_speech_commands.pth")
        print(" Modèle sauvegardé : simple_cnn_speech_commands.pth")

    except FileNotFoundError as fnf_error:
        print(f"Erreur de fichier : {fnf_error}")

    except ValueError as val_error:
        print(f"Erreur de valeur : {val_error}")

    except RuntimeError as rt_error:
        print(f"Erreur d'exécution : {rt_error}")

    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == "__main__":
    main()