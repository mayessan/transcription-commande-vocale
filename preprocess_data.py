import re

# Expression régulière définissant les caractères à ignorer/supprimer :
# Elle inclut : virgules, ?, ., !, -, ;, :, ", %, ‘, ”, � (caractère inconnu)
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

#  Fonction de nettoyage : supprime les caractères spéciaux d'une phrase
def remove_special_characters(batch):
    # Accès au texte original via batch["sentence"]
    # ➕ Supprime les caractères définis par l'expression regex
    # ➕ Convertit en minuscules
    # ➕ Ajoute un espace à la fin 
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch