# Corriger le `NaN` en sortie sur ESP32

## Objectif

Le firmware démarre maintenant, alloue l’interpréteur TensorFlow Lite Micro, capture l’audio, extrait les features et appelle le modèle. Le problème restant est que la probabilité finale peut devenir `NaN` sur l’ESP32.

Ce document liste les vérifications à faire dans le bon ordre, à partir des correctifs déjà tentés dans ce dépôt.

## Ce qu’on a déjà essayé

- L’API publique MFCC a été simplifiée et les helpers internes ont été séparés.
- Les plus gros buffers statiques MFCC ont été retirés de la RAM globale.
- L’extraction des features utilise maintenant des buffers temporaires sur le heap au lieu de grosses matrices persistantes.
- Les statistiques cumulées ont remplacé le stockage de tout l’historique pour les résumés.
- L’arène TensorFlow Lite a été réduite pour mieux tenir sur la carte.
- Le firmware a maintenant des logs de debug dédiés pour l’audio, les features, les métadonnées des tenseurs et la sortie du modèle.
- Des tests d’inférence avec vecteur nul et vecteur plein de `1` ont été ajoutés.
- Le pipeline Python d’entraînement a été vérifié via les rapports Keras et TFLite.
- La chaîne d’export TFLite a été vérifiée et enregistre maintenant un SHA-256 du `.tflite` source dans le `inference.cpp` généré.

## Faits actuels

- La validation côté entraînement indique que le fichier `.tflite` est sain.
- L’ESP32 renvoie toujours `NaN`, même sur des entrées synthétiques comme tout à `0` et tout à `1`.
- La capture audio n’est plus le suspect principal.
- L’extraction des features n’est plus le suspect principal, parce que le chemin de debug prouve déjà que le vecteur de sortie est fini avant l’inférence.
- Le problème le plus probable restant se situe dans l’intégration du modèle dans le firmware ou dans le comportement runtime de TFLite Micro sur ESP32.

## Théories classées par probabilité

### 1. Le firmware n’exécute pas exactement le même artefact modèle que celui validé sur PC

Pourquoi c’est plausible :

- La validation PC lit directement le `.tflite`.
- Le firmware utilise le blob généré dans `inference.cpp`.
- Un fichier généré obsolète ou décalé peut conserver un bon rapport d’entraînement tout en cassant le runtime ESP32.

Ce qu’il faut vérifier :

- Régénérer `firmware/src/model/inference.cpp` à partir du `.tflite` actuel.
- Vérifier le commentaire SHA-256 ajouté par le générateur.
- Comparer le hash du `.tflite` source avec les métadonnées du blob généré.
- Vérifier que le firmware est recompilé après la régénération et pas encore basé sur un ancien `inference.cpp`.

### 2. TFLite Micro tourne, mais les hypothèses sur l’entrée ou la sortie du modèle sont fausses sur ESP32

Pourquoi c’est plausible :

- Le modèle peut toujours allouer les tenseurs et exposer `input(0)` / `output(0)`.
- Un décalage de forme, de type ou de quantization peut encore produire un calcul invalide ou une sortie indéfinie à l’exécution.

Ce qu’il faut vérifier :

- Afficher le type des tenseurs d’entrée et de sortie dans `main.cpp`.
- Vérifier que le tenseur d’entrée est bien en `float32` si le firmware écrit des features en `float32`.
- Vérifier que la forme du tenseur d’entrée correspond exactement à `FEATURE_VECTOR_SIZE`.
- Vérifier que le tenseur de sortie est bien une probabilité scalaire.
- Vérifier que `input->bytes` vaut `FEATURE_VECTOR_SIZE * sizeof(float)`.
- Vérifier que `output->data.f[0]` est fini juste après `Invoke()`.

### 3. Le blob modèle est correct, mais le runtime ESP32 ou l’état mémoire corrompt l’inférence

Pourquoi c’est plausible :

- Un `NaN` sur des entrées synthétiques est un symptôme runtime, pas un symptôme d’entraînement.
- Si le modèle est correct sur PC, les problèmes mémoire spécifiques à l’ESP32 deviennent plus probables.

Ce qu’il faut vérifier :

- Lancer le test sur vecteur nul.
- Lancer le test sur vecteur plein de `1`.
- Si les deux renvoient déjà `NaN`, le problème est en dessous de l’extracteur de features.
- Vérifier que `tensor_arena` n’est pas trop petite pour le modèle réel.
- Vérifier l’absence d’écriture hors limites dans le code qui touche le buffer d’entrée.
- Vérifier l’absence de corruption stack/heap autour de l’appel d’inférence.

### 4. Le vecteur de features est fini, mais son échelle n’est pas celle attendue par le modèle

Pourquoi c’est moins probable maintenant, mais encore à vérifier :

- Le chemin de debug a déjà montré que le vecteur existe et qu’il est fini.
- En revanche, un décalage de distribution des features peut encore rendre le modèle instable ou sans sens.

Ce qu’il faut vérifier :

- Comparer la plage des features sur ESP32 avec le cache d’entraînement.
- Vérifier le min, le max, la moyenne et l’écart-type des features exportées.
- Vérifier que la même normalisation est utilisée côté PC et côté ESP32.
- Vérifier que l’entraînement et le firmware utilisent les mêmes valeurs de `FeatureConfig`.

### 5. Le pipeline DSP a un problème numérique caché qui n’apparaît que sur l’exécution embarquée

Pourquoi c’est possible :

- Le code flottant sur ESP32 peut se comporter différemment de Python sur desktop.
- Une division par zéro, un logarithme invalide ou une normalisation mal protégée peuvent créer un `NaN` sans bruit.

Ce qu’il faut vérifier :

- Inspecter chaque logarithme, chaque division et chaque normalisation dans l’extraction MFCC.
- Vérifier que le calcul de l’énergie Mel et du log-Mel clamp bien les valeurs nulles ou négatives avant `log`.
- Vérifier que les calculs delta et delta2 ne divisent jamais par zéro.
- Vérifier que tous les résumés de features gèrent correctement les fenêtres vides ou dégénérées.

## Checklist concrète, dans l’ordre

### Étape 1. Confirmer que l’artefact modèle généré est à jour

- Régénérer `inference.cpp` à partir du dernier `.tflite`.
- Confirmer que le commentaire SHA-256 correspond bien au modèle source actuel.
- Supprimer l’ancien fichier généré en cas de doute, puis régénérer.
- Recompiler le firmware après la régénération.

### Étape 2. Confirmer que le firmware voit le contrat tensoriel attendu

- Logguer le type du tenseur d’entrée.
- Logguer le type du tenseur de sortie.
- Logguer les dimensions du tenseur d’entrée.
- Logguer les dimensions du tenseur de sortie.
- Logguer la taille des bytes d’entrée.
- Logguer la taille des bytes de sortie.
- Vérifier que l’entrée reste bien `1 x FEATURE_VECTOR_SIZE`.

### Étape 3. Reproduire le `NaN` uniquement avec des vecteurs synthétiques

- Injecter un vecteur nul.
- Injecter un vecteur rempli de `1`.
- Injecter un vecteur de features connu comme bon, copié depuis un échantillon d’entraînement.
- Comparer les trois sorties.

Interprétation :

- Si le vecteur nul et le vecteur `1` renvoient déjà `NaN`, le bug est dans la couche modèle/runtime.
- Si seules les vraies features renvoient `NaN`, le bug est plus probablement dans le chemin d’extraction des features.

### Étape 4. Vérifier les risques mémoire et les corruptions

- Confirmer que `arena_used_bytes()` est cohérent après `AllocateTensors()`.
- Confirmer qu’il n’y a aucun dépassement de buffer dans les tableaux audio ou feature.
- Confirmer que les allocations temporaires réussissent pendant l’extraction des features.
- Confirmer qu’aucune fonction n’écrit au-delà de `FEATURE_VECTOR_SIZE`.

### Étape 5. Comparer les features firmware avec les features d’entraînement

- Afficher les premières valeurs du vecteur de features firmware.
- Les comparer avec le cache de features d’entraînement.
- Comparer la moyenne, le minimum, le maximum et la variance.
- Vérifier que l’ordre des features est strictement identique.

### Étape 6. Valider le `.cpp` généré et le `.tflite` d’entraînement entre eux

- Vérifier les métadonnées SHA ajoutées par le générateur.
- Vérifier la taille du fichier.
- Vérifier les premiers octets de l’en-tête du blob.
- Confirmer que la même révision du modèle est utilisée dans les rapports d’entraînement et dans la génération firmware.

### Étape 7. Si tout le reste est correct, isoler TFLite Micro lui-même

- Construire un test firmware minimal qui charge seulement le modèle et lance un vecteur nul.
- Retirer la capture audio du test.
- Retirer l’extraction de features du test.
- Si ce test minimal renvoie encore `NaN`, le problème est dans l’intégration TFLite Micro ou dans la compatibilité du modèle sur ESP32.

## Conclusion la plus probable

À partir de tout ce qui a déjà été essayé, le diagnostic le plus probable est :

1. Le `.tflite` côté entraînement est bon.
2. Le `inference.cpp` généré est probablement bon s’il a bien été régénéré depuis le bon artefact.
3. Le `NaN` est le plus probablement causé par la partie runtime sur ESP32 : décalage de blob modèle, décalage du contrat tensoriel, corruption mémoire ou incompatibilité TFLite Micro.

## À faire si le bug persiste

- Ajouter un diagnostic minimal qui contourne complètement l’audio et les features.
- Comparer le hash du blob généré par le firmware avec le hash du `.tflite` source.
- Si les hashes correspondent et que les entrées synthétiques donnent encore `NaN`, réduire le problème à un repro TFLite Micro pur sur ESP32.
