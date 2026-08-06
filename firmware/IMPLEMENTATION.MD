# Firmware Implementation

## Objectif

Cette implémentation firmware enchaîne 3 étapes sur l'ESP32:

1. enregistrer 2 secondes d'audio via I2S,
2. extraire les features audio dans le même esprit que l'entraînement Python,
3. lancer l'inférence TFLite et afficher sur `Serial` si une scie est détectée.

L'idée est de garder le firmware aligné avec la configuration d'entraînement:
- `sample_rate = 8000`
- `window_seconds = 2.0`
- `hop_seconds = 1.0`
- `n_mfcc = 20`
- `n_mels = 32`
- `fft_length = 1024`
- `hop_length = 256`

## Ce qui a été implémenté

### `firmware/src/main.cpp`

#### `setup()`

Cette fonction initialise tout le runtime embarqué:

- démarre le port `Serial` à `115200`,
- charge le modèle TFLite embarqué (`g_model_data`),
- instancie l'interpréteur `MicroInterpreter`,
- alloue les tenseurs TFLite,
- initialise l'entrée audio I2S avec `init_audio()`,
- prépare l'extracteur de features avec `init_feature_extractor()`,
- affiche des infos de debug sur le terminal: taille de l'arène mémoire, taille du vecteur de features et seuil de détection.

#### `loop()`

Cette fonction fait la boucle de détection en continu:

1. elle enregistre l'audio dans `audio_buffer` avec `record_audio()`,
2. elle transforme le signal en vecteur de features avec `extract_features_from_audio()`,
3. elle copie ce vecteur dans le tenseur d'entrée du modèle,
4. elle lance `interpreter->Invoke()`,
5. elle lit la probabilité de sortie,
6. elle compare cette probabilité au seuil avec `is_chainsaw_detected()`,
7. elle affiche le verdict sur `Serial`.

Sortie typique:

```text
Probabilite tronconneuse: 0.873421 | verdict: SCIE DETECTEE
```

### `firmware/src/services/mfcc.cpp`

Ce fichier contient le cœur du pipeline DSP embarqué.

#### `init_feature_extractor()`

Prépare les coefficients utilisés pour les dérivées temporelles:

- calcule une fois les coefficients de `delta` (première dérivée),
- calcule une fois les coefficients de `delta2` (seconde dérivée),
- évite de refaire ce travail à chaque boucle audio.

#### `extract_features_from_audio(...)`

C'est la fonction la plus importante du firmware. Elle prend le buffer audio brut et fabrique un vecteur de `194` features.

Pipeline effectué:

1. **normalisation du signal**
   - conversion `int16_t` vers `float`,
   - normalisation par le pic local du signal.

2. **fenêtrage temporel**
   - fenêtre de Hann,
   - FFT de taille `1024`,
   - saut de `256` échantillons.

3. **spectre**
   - calcul du module de la FFT,
   - extraction des bins fréquentiels utiles.

4. **banc Mel**
   - projection sur `32` bandes Mel,
   - réutilisation de la matrice `MEL_MATRIX` définie dans `firmware/src/app/dsp_constants.cpp`.

5. **compression logarithmique**
   - conversion en dB,
   - plafonnement dynamique avec un top d'environ `80 dB`.

6. **MFCC**
   - projection DCT via `DCT_MATRIX`,
   - production de `20` coefficients MFCC par trame.

7. **delta et delta2**
   - calcul des dérivées temporelles sur les MFCC.

8. **résumés statistiques**
   - moyenne et écart-type pour chaque série temporelle,
   - concaténation finale pour obtenir un vecteur plat de `194` valeurs.

Le format final suit le modèle entraîné, qui attend un vecteur de `194` floats.

#### `is_chainsaw_detected(float probability, float threshold)`

Simple fonction de décision:

- renvoie `true` si `probability >= threshold`,
- renvoie `false` sinon.

### `firmware/include/services/mfcc.h`

Ce header expose les constantes et les fonctions DSP:

- `AUDIO_WINDOW_SAMPLES = 16000`
- `FEATURE_FRAME_COUNT = 63`
- `FEATURE_VECTOR_SIZE = 194`
- `DETECTION_THRESHOLD = 0.5f`

Fonctions déclarées:

- `init_feature_extractor()`
- `extract_features_from_audio(...)`
- `is_chainsaw_detected(...)`

### `firmware/include/model/inference.h`

Expose le tableau du modèle TFLite embarqué:

- `g_model_data[]`

Ce tableau est utilisé par `tflite::GetModel(...)` dans `main.cpp`.

### `firmware/include/app/config.h`

Centralise la configuration audio commune au firmware et au pipeline d'entraînement:

- fréquence d'échantillonnage,
- longueur de fenêtre,
- hop,
- taille FFT,
- nombre de MFCC,
- nombre de bandes Mel.

## Comment ça fonctionne à l'exécution

1. Le micro INMP441 est lu en I2S.
2. Le firmware enregistre une fenêtre de `2` secondes.
3. Les features sont calculées localement sur l'ESP32.
4. Le vecteur est injecté dans le modèle TFLite Micro.
5. Le modèle retourne une probabilité.
6. Le firmware affiche si la scie est détectée ou non.

## Comment tester

### 1. Vérifier le câblage

Le micro INMP441 doit être câblé correctement:

- `SCK` -> `GPIO 14`
- `WS` -> `GPIO 15`
- `SD` -> `GPIO 32`
- `L/R` -> `GND`

Le point `L/R` à la masse est important pour que le canal utilisé corresponde bien à la config du code.

### 2. Compiler le firmware

Depuis la racine du projet:

```bash
platformio run -d firmware
```

### 3. Flasher l'ESP32

```bash
platformio run -d firmware --target upload
```

### 4. Ouvrir le moniteur série

```bash
platformio device monitor -p COMx -b 115200
```

Sur Windows, tu peux aussi utiliser le moniteur PlatformIO depuis VS Code.

### 5. Ce qu'il faut voir au démarrage

Au boot, le terminal doit afficher des infos du genre:

- initialisation de l'IA,
- modèle chargé,
- arène mémoire utilisée,
- taille du vecteur de features,
- seuil de détection.

Ensuite, à chaque fenêtre audio, tu dois voir quelque chose comme:

```text
Probabilite tronconneuse: 0.812345 | verdict: SCIE DETECTEE
```

ou

```text
Probabilite tronconneuse: 0.143210 | verdict: PAS DE SCIE
```

## Points d'attention

- Le vecteur de features doit rester à `194` valeurs, sinon l'entrée du modèle ne correspondra plus.
- Si le modèle Python change, il faut vérifier que le firmware produit toujours les mêmes dimensions et le même ordre de features.
- Si le build devient trop lourd, `TensorFlowLite_ESP32` peut demander un ajustement de l'arène mémoire ou des libs utilisées.

## Résumé

Le firmware fait maintenant la chaîne complète:

- capture audio,
- extraction DSP locale,
- inférence TFLite,
- affichage du verdict sur `Serial`.

Le point clé est que l'extraction est structurée pour rester compatible avec la configuration d'entraînement du projet.
