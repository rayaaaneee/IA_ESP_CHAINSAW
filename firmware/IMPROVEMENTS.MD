## Résumé du changement

## L'IA détecte bien les scies mais bcp de faux positifs. La latence de détection est trop longue (4s) et le modèle n'est pas encore assez robuste pour la jungle colombienne. Une augmentation de données est nécessaire pour améliorer la robustesse du modèle.

Dans main.cpp, la fenêtre audio de 2s (`AUDIO_WINDOW_SAMPLES`) est maintenant **glissante** au lieu d'être recapturée entièrement à chaque tour :
- Premier passage : capture complète des 2s (comme avant).
- Passages suivants : le buffer est décalé (`memmove`) pour jeter la plus ancienne seconde (`kHopSamples = 8000` échantillons = `hop_seconds`), puis seule **1 nouvelle seconde** est lue via I2S et ajoutée à la fin.

Résultat : une inférence toutes les **~1s** au lieu de **~2s**, ce qui réduit le pire cas de latence de détection d'environ 4s à environ 2s (le temps qu'une fenêtre complète soit majoritairement dominée par le bruit de la tronçonneuse).

Flash et teste à nouveau — la détection devrait réagir plus vite. Si tu veux réduire encore plus la latence, on peut baisser `hop_seconds` (ex: 0.5s) dans config.h et régénérer, mais cela augmente la fréquence d'inférence (donc la charge CPU) en échange d'une réactivité accrue.