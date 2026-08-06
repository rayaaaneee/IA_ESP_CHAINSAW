# Resolving Notes

Ce document détaille l'ensemble des modifications appliquées au projet, avec un focus particulier sur le firmware ESP32, le pipeline MFCC, la séparation des responsabilités entre fichiers, et les corrections mémoire qui ont permis de faire passer la compilation.

## Résumé exécutif

Le firmware a été restructuré pour suivre un découpage plus propre:

- `mfcc.h` expose uniquement l'API publique.
- `mfcc_utils.h` / `mfcc_utils.cpp` portent les helpers internes et les coefficients de dérivation.
- `main.cpp` orchestre la capture audio, l'extraction de features et l'inférence TFLite.
- La consommation DRAM a été réduite en supprimant les gros buffers persistants du pipeline MFCC.

Le résultat final est un firmware compilable qui conserve le même objectif fonctionnel: enregistrer l'audio, extraire un vecteur de 194 features, lancer l'inférence et afficher un verdict sur `Serial`.

## Ce qui a changé globalement

### 1. Public API vs implémentation interne

Le code MFCC a été séparé en deux couches:

- `firmware/include/services/mfcc.h` contient uniquement les constantes publiques et les trois fonctions d'usage:
  - `init_feature_extractor()`
  - `extract_features_from_audio(...)`
  - `is_chainsaw_detected(...)`

- `firmware/include/services/mfcc_utils.h` et `firmware/src/services/mfcc_utils.cpp` contiennent les détails de calcul:
  - fonctions mathématiques internes,
  - calcul des coefficients `delta` et `delta2`,
  - helpers de réflexion de signal,
  - initialisation du pipeline interne.

L'objectif était de garder le header public minimal et d'éviter d'exposer au reste du firmware des détails de DSP qui n'ont pas besoin d'être connus en dehors du module.

### 2. Réduction mémoire du pipeline MFCC

Le problème principal rencontré était un overflow linker de `.dram0.bss` sur ESP32.

Les gros tableaux globaux qui stockaient les buffers intermédiaires du pipeline ont été supprimés de la mémoire statique:

- `g_signal`
- `g_frame_real`
- `g_frame_imag`
- `g_magnitude`
- `g_mel_energies`
- `g_mel_db`
- `g_mfcc_matrix`
- `g_delta_matrix`
- `g_delta2_matrix`
- `g_centroid`
- `g_bandwidth`
- `g_rolloff`
- `g_zcr`
- `g_rms`

À la place, le calcul utilise:

- des buffers temporaires alloués au moment de l'extraction,
- des accumulateurs `RunningStats` pour calculer moyenne et écart-type en streaming,
- des tableaux locaux réutilisables pour éviter la conservation de matrices complètes en RAM.

Le but était de libérer suffisamment de DRAM pour que le firmware puisse linker sur `nodemcu-32s`.

### 3. Ajustement du `tensor_arena`

Dans `firmware/src/main.cpp`, la taille du `tensor_arena` a été réduite:

- anciennement `50 * 1024`
- maintenant `24 * 1024`

Ce changement a été fait parce que l'arène TFLite prenait une part importante de la DRAM disponible, en concurrence directe avec les buffers audio et DSP.

### 4. Nettoyage des imports

Plusieurs imports ont été simplifiés ou déplacés pour rendre le graphe d'inclusion plus sain:

- `mfcc.h` ne dépend plus des helpers internes.
- `mfcc_utils.h` ne dépend pas de `mfcc.h`.
- les constantes partagées restent dans `mfcc_config.h`.
- les constantes DSP de Mel/DCT restent dans `app/dsp_constants.h`.

L'objectif était d'éviter les dépendances transverses inutiles et de limiter les effets de bord au moment de la compilation.

## Détail par fichier

### `firmware/src/main.cpp`

Rôle: orchestrateur principal du firmware.

Changements notables:

- initialisation du `Serial` et du modèle TFLite,
- création de l'interpréteur micro,
- réduction de la taille de `tensor_arena` à `24 KB`,
- capture d'un buffer audio de `AUDIO_WINDOW_SAMPLES`,
- appel à `extract_features_from_audio(...)`,
- vérification de la taille d'entrée du tenseur,
- copie des features dans le tenseur TFLite,
- invocation du modèle,
- affichage du verdict en anglais sur `Serial`.

Messages de log:

- `Initializing the AI...`
- `AI model loaded and ready for inference.`
- `Error: unable to allocate tensors!`
- `Error: unable to read audio.`
- `Error: unable to extract features.`
- `Error: unexpected input size...`
- `Error: inference failed!`
- `Chainsaw probability: ... | verdict: ...`

Le fichier conserve un petit forward declaration de `extract_features_from_audio(...)` pour éviter toute ambiguïté de résolution pendant l'édition.

### `firmware/src/services/mfcc.cpp`

Rôle: exécuter l'extraction de features publique.

Changements notables:

- le fichier garde seulement l'API publique du module MFCC,
- l'extraction a été réécrite pour ne plus dépendre de grosses matrices persistantes,
- le calcul s'appuie maintenant sur:
  - des buffers heap temporaires (`frame_real`, `frame_imag`, `mel_energies`, `mfcc_matrix`),
  - des `RunningStats` pour calculer les statistiques finales,
  - les matrices DSP précomputées `MEL_MATRIX` et `DCT_MATRIX`,
  - les fonctions utilitaires de `mfcc_utils`.

Fonctions publiques maintenues:

- `extract_features_from_audio(...)`
- `init_feature_extractor()`
- `is_chainsaw_detected(...)`

Ce fichier fait la transition entre:

- le signal brut 16-bit,
- et les 194 valeurs finales attendues par le modèle.

### `firmware/src/services/mfcc_utils.cpp`

Rôle: helpers internes du pipeline DSP.

Changements notables:

- conservation uniquement des coefficients `delta` et `delta2`,
- conservation du booléen `g_initialized`,
- fonctions internes mathématiques et de réflexion du signal,
- suppression de toutes les matrices de travail volumineuses,
- maintien de la génération des coefficients dérivés au démarrage.

Fonctions internes importantes:

- `hann_window(...)`
- `reflect_index(...)`
- `sample_reflect(...)`
- `invert_2x2(...)`
- `invert_3x3(...)`
- `build_delta_coefficients(...)`
- `compute_delta_series(...)`
- `ensure_initialized()`

Le rôle de ce fichier est devenu strictement utilitaire.

### `firmware/include/services/mfcc.h`

Rôle: API publique du module MFCC.

Contenu conservé:

- `AUDIO_WINDOW_SAMPLES`
- `FEATURE_FRAME_COUNT`
- `FEATURE_VECTOR_SIZE`
- `DETECTION_THRESHOLD`
- déclaration des fonctions publiques du module

Ce header a été volontairement gardé léger pour éviter d'exposer les helpers internes au reste du firmware.

### `firmware/include/services/mfcc_utils.h`

Rôle: contrat des helpers internes.

Contenu actuel:

- constantes internes du pipeline MFCC,
- déclaration de `g_delta_coeffs` et `g_delta2_coeffs`,
- déclaration des helpers internes de calcul.

Ce header est utilisé par `mfcc.cpp` et `mfcc_utils.cpp`, mais il n'a pas vocation à être consommé directement par d'autres modules.

### `firmware/include/services/mfcc_config.h`

Rôle: compatibilité / configuration partagée.

Ce fichier contient les constantes partagées par le module MFCC:

- `AUDIO_WINDOW_SAMPLES = 16000`
- `FEATURE_FRAME_COUNT = 63`
- `FEATURE_VECTOR_SIZE = 194`
- `DETECTION_THRESHOLD = 0.5f`

Il sert de point d'ancrage pour les valeurs publiques utilisées dans le pipeline.

### `firmware/include/app/dsp_constants.h`

Rôle: constantes DSP précomputées.

Ce fichier expose:

- `MEL_MATRIX`
- `DCT_MATRIX`

avec leurs tailles calculées à partir de `AUDIO_CONFIG`.

C'est la base mathématique du calcul des coefficients Mel et MFCC.

### `firmware/src/model/inference.cpp` et `firmware/include/model/inference.h`

Rôle: embarquer le modèle TFLite sous forme de tableau binaire.

Le header expose:

- `g_model_data[]`
- `g_model_data_len`

Le `.cpp` contient les octets du modèle.

Le firmware charge ce tableau via `tflite::GetModel(g_model_data)`.

### `firmware/src/drivers/audio.cpp`

Rôle: configurer l'entrée audio I2S et lire les échantillons.

Il initialise l'I2S avec:

- `I2S_MODE_MASTER | I2S_MODE_RX`
- fréquence définie par `AUDIO_CONFIG.sample_rate`
- `I2S_CHANNEL_FMT_ONLY_LEFT`

La fonction `record_audio(...)` lit un buffer complet de `int16_t`.

Le câblage INMP441 attendu est cohérent avec la configuration matérielle du projet.

## Réduction mémoire: pourquoi cette refonte était nécessaire

Le linker error rencontré était:

```text
section `.dram0.bss' will not fit in region `dram0_0_seg'
region `dram0_0_seg' overflowed by 93744 bytes
```

Cette erreur ne venait pas d'une faute logique, mais d'une saturation de la RAM interne du ESP32.

Les gros buffers MFCC ont donc été traités de façon prioritaire:

1. supprimer les statics lourds,
2. passer sur des buffers temporaires,
3. calculer les statistiques en streaming,
4. réduire le `tensor_arena`.

## Résultat fonctionnel attendu

Le firmware fait désormais la chaîne suivante:

1. initialisation du micro et du modèle,
2. capture de `2` secondes d'audio,
3. extraction locale des features,
4. vérification de la taille de l'entrée modèle,
5. inférence TFLite,
6. affichage du verdict `CHAINSAW DETECTED` ou `NO CHAINSAW`.

## Vérifications effectuées

### Vérification statique

Les fichiers principaux ont été vérifiés sans erreur par l'analyseur:

- `firmware/src/main.cpp`
- `firmware/src/services/mfcc.cpp`
- `firmware/src/services/mfcc_utils.cpp`
- `firmware/include/services/mfcc.h`
- `firmware/include/services/mfcc_utils.h`

### Vérification build

La compilation PlatformIO a finalement passé avec `inv build`.

Cela confirme que le firmware linke correctement après la réduction mémoire.

## Ce que j'ai changé sur les imports

Voici le point précis demandé sur les imports:

- j'ai retiré les includes internes du header public `mfcc.h`,
- j'ai déplacé les dépendances de détail vers `mfcc_utils.h`,
- j'ai limité `mfcc_utils.h` aux helpers du module,
- j'ai gardé `app/dsp_constants.h` là où les matrices Mel et DCT sont réellement utilisées,
- j'ai évité de propager les dépendances DSP jusqu'au code d'appel du firmware.

Autrement dit: le noyau d'exécution reste dans `main.cpp`, le calcul dans `mfcc.cpp`, et le détail mathématique dans `mfcc_utils.cpp`.

## Ce que j'ai changé sur les fonctions

Fonctions ajoutées ou conservées:

- `init_feature_extractor()`
- `extract_features_from_audio(...)`
- `is_chainsaw_detected(...)`
- `ensure_initialized()`
- `build_delta_coefficients(...)`
- `compute_delta_series(...)`
- `reflect_index(...)`
- `sample_reflect(...)`
- `invert_2x2(...)`
- `invert_3x3(...)`
- `hann_window(...)`

Fonctions importantes introduites localement dans `mfcc.cpp`:

- `RunningStats::add(...)`
- `RunningStats::mean(...)`
- `RunningStats::stddev(...)`
- `write_stats(...)`

Ces fonctions servent à calculer le vecteur final sans conserver tous les buffers intermédiaires en mémoire.

## Conclusion

Le projet a été modifié pour tenir à la fois sur le plan architectural et sur le plan mémoire.

Les changements les plus importants sont:

- séparation nette entre API publique et helpers internes,
- allègement massif de la RAM statique,
- réduction du `tensor_arena`,
- conservation du pipeline audio / DSP / TFLite,
- build désormais valide.

Ce document sert de trace de résolution: il explique à la fois ce qui a été changé, pourquoi, et quel symptôme concret cela a corrigé.
