## Priorité 1 - firmware ESP32
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
