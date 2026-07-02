# TODO

## Priorité 1 - chaîne IA
- [ ] Définir le pipeline audio de bout en bout.
- [ ] Fixer le format d'entrée: fichiers WAV, fréquence d'échantillonnage, durée de fenêtre, mono/stéréo, normalisation.
- [ ] Définir les classes cibles: `CHAINSAW` et `EVERYTHING ELSE`.
- [ ] Organiser `training/dataset/` en structure stable pour l'entraînement et la validation.
- [ ] Implémenter `training/extract_features.py` pour charger les WAV, découper les fenêtres et extraire les features.
- [ ] Choisir les features de départ: spectrogramme, MFCC ou combinaison simple et robuste.
- [ ] Ajouter la gestion des labels et du split train/validation/test.
- [ ] Implémenter `training/ai_model.py` avec une vraie architecture de classification binaire.
- [ ] Prévoir les méthodes minimales: `train`, `predict`, `evaluate`, `save_model`, `load_model`.
- [ ] Écrire `training/train.py` pour orchestrer le chargement des données, l'entraînement et la sauvegarde.
- [ ] Ajouter des métriques utiles: accuracy, loss, precision, recall et matrice de confusion.
- [ ] Vérifier que le modèle est sauvegardé dans un format exploitable par l'étape de conversion.
- [ ] Valider que le modèle atteint un niveau de performance suffisant avant export vers l'ESP32.

## Priorité 2 - conversion embarquée
- [ ] Corriger `training/convert_to_tflite.py` pour pointer vers les bons chemins.
- [ ] Générer proprement `model.tflite` et `model.h` dans un dossier cohérent avec le firmware.
- [ ] Vérifier la compatibilité du modèle TFLite avec les contraintes mémoire de l'ESP32.
- [ ] Ajouter si besoin une quantification adaptée au TinyML.

## Priorité 3 - firmware ESP32
- [ ] Remplacer le blink LED de `firmware/src/main.cpp` par la logique applicative réelle.
- [ ] Intégrer le modèle embarqué dans le firmware.
- [ ] Implémenter l'acquisition du signal micro / capteur.
- [ ] Ajouter le prétraitement côté ESP32 avant inférence.
- [ ] Définir la logique de décision et d'action en sortie de prédiction.
- [ ] Nettoyer ou compléter les templates dans `firmware/src/templates/`.

## Priorité 4 - outillage et automatisation
- [ ] Vérifier les tâches `invoke` dans `tasks.py`.
- [ ] Aligner les chemins entre training, conversion et firmware.
- [ ] S'assurer que `requirements.in` et `requirements.txt` couvrent toutes les dépendances.
- [ ] Tester les scripts d'initialisation `init.ps1`, `init.bat` et `init.sh`.
- [ ] Vérifier que PlatformIO est bien configuré pour le board cible.

## Priorité 5 - validation et documentation
- [ ] Ajouter un jeu de tests ou de scripts de validation pour le pipeline IA.
- [ ] Documenter le flux complet: collecte des données, entraînement, conversion, flash.
- [ ] Clarifier la structure du workspace et des dossiers du projet.
- [ ] Nettoyer les artefacts générés et les chemins dupliqués si nécessaire.

## Questions ouvertes
- [ ] Quel capteur audio est utilisé exactement ?
- [ ] Quel est le format attendu des données d'entraînement ? WAV
- [ ] Quelle classe finale doit être détectée par le modèle ? CHAINSAW / EVERYTHING ELSE
- [ ] Faut-il une détection temps réel ou par fenêtre glissante ?
