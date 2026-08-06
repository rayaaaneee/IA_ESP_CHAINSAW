# Résolution du débordement DRAM

## Contexte

Le firmware échouait initialement à l’édition de liens sur l’ESP32 avec cette erreur :

```text
section `.dram0.bss' will not fit in region `dram0_0_seg'
region `dram0_0_seg' overflowed by 93744 bytes
```

Ce n’était pas un bug logique dans le pipeline MFCC lui-même. C’était un problème de placement mémoire : trop de RAM statique était réservée dans `.bss`, principalement à cause de gros tableaux persistants utilisés par l’extraction de features et par l’arène mémoire TFLite.

L’objectif de la correction était de libérer assez de DRAM tout en conservant le même comportement à l’exécution et le même format de 194 features attendu par le modèle.

## Ce qui a été modifié

### 1. L’en-tête de l’API firmware a été simplifié

`firmware/include/services/mfcc.h` a été réduit à la seule API publique :

- `init_feature_extractor()`
- `extract_features_from_audio(...)`
- `is_chainsaw_detected(...)`

Il ne conserve plus que les constantes publiques partagées via `mfcc_config.h`, au lieu d’embarquer l’en-tête interne des helpers.

C’est important, car l’en-tête public ne doit pas exposer les détails d’implémentation du pipeline DSP.

### 2. Un en-tête dédié a été conservé pour les utilitaires internes MFCC

`firmware/include/services/mfcc_utils.h` contient maintenant uniquement les déclarations internes utilisées par l’implémentation MFCC :

- les helpers des coefficients delta,
- les helpers de réflexion/padding,
- les helpers d’inversion de matrices,
- la génération des séries delta,
- l’aide à l’initialisation.

Ce fichier est destiné aux fichiers d’implémentation, pas à l’API externe.

### 3. Les gros buffers persistants de features ont été retirés de la RAM statique

La plus grosse source de pression sur `.dram0.bss` était `mfcc_utils.cpp`.

Avant le refactor, il contenait de nombreux tableaux globaux volumineux comme :

- le stockage du signal audio,
- les buffers de travail FFT,
- les buffers de magnitude,
- les matrices d’énergie Mel,
- les matrices MFCC,
- les matrices delta,
- les buffers centroid / bandwidth / rolloff / ZCR / RMS.

Ces tableaux ont été retirés du stockage statique.

Il ne reste dans `mfcc_utils.cpp` que les petites données persistantes réellement nécessaires :

- `g_delta_coeffs`
- `g_delta2_coeffs`
- `g_initialized`

Le reste de l’état d’extraction est désormais temporaire et alloué uniquement pendant l’exécution de l’extraction.

### 4. La fonction d’extraction a été réécrite pour utiliser des buffers temporaires

`firmware/src/services/mfcc.cpp` a été réécrit pour ne plus dépendre de matrices globales persistantes.

La nouvelle approche est :

- allocation de buffers temporaires FFT et features sur le heap,
- calcul des features trame par trame,
- accumulation progressive des statistiques,
- libération des buffers temporaires avant le retour.

Cela évite de garder de grosses matrices intermédiaires en mémoire pendant toute la durée de vie du firmware.

### 5. Les statistiques de features sont passées d’un stockage complet à une agrégation en continu

Au lieu de stocker tout l’historique des features dans des tableaux globaux puis de les résumer ensuite, le code utilise maintenant une structure locale `RunningStats`.

Cette structure suit :

- la somme,
- la somme des carrés,

pour chaque flux de features.

Le firmware peut ainsi calculer :

- la moyenne,
- l’écart-type,

sans devoir stocker tous les vecteurs intermédiaires en RAM statique.

### 6. Delta et delta2 ont été conservés, mais uniquement comme buffers de calcul temporaires

Le pipeline delta a été conservé, mais son modèle de stockage a changé.

Le firmware calcule toujours :

- les features MFCC,
- les features delta MFCC,
- les features delta2 MFCC,
- les features de résumé Mel,
- les résumés centroid / bandwidth / rolloff / ZCR / RMS.

Cependant, les matrices intermédiaires sont maintenant temporaires au lieu d’être globales.

### 7. L’arène TFLite a été réduite

Dans `firmware/src/main.cpp`, l’arène mémoire TFLite a été réduite :

```cpp
constexpr int kTensorArenaSize = 24 * 1024;
```

C’est une partie majeure de la correction, car l’arène consomme elle aussi de la RAM interne.

Une arène plus petite donne davantage d’espace au linker et réduit l’empreinte statique.

### 8. Les imports ont été nettoyés pour correspondre à la nouvelle architecture

Comme l’implémentation MFCC a changé, les includes ont aussi été ajustés.

Points importants :

- `mfcc.h` n’inclut plus l’en-tête interne des helpers.
- `mfcc_utils.h` n’expose que les déclarations des helpers, pas l’API publique.
- `mfcc.cpp` n’inclut plus que ce dont il a besoin pour faire l’extraction.

Cela clarifie la séparation et évite de traîner les dépendances d’implémentation dans l’interface publique.

## Pourquoi ces changements corrigent le débordement

L’erreur du linker venait d’une saturation de la DRAM statique, pas du flot d’exécution.

La correction fonctionne parce qu’elle retire les gros tableaux statiques de `.bss` et les remplace par :

- des allocations temporaires sur le heap,
- des statistiques en flux,
- moins d’état persistant,
- une arène TFLite plus petite.

C’est exactement le type de changement qui libère les dizaines de kilo-octets demandés par le linker.

## Fichiers touchés et rôle de chacun

### `firmware/include/services/mfcc.h`

API publique MFCC uniquement.

### `firmware/include/services/mfcc_utils.h`

Déclarations internes des helpers MFCC.

### `firmware/src/services/mfcc_utils.cpp`

Petit état persistant uniquement :

- coefficients delta,
- initialisation des coefficients,
- helpers de réflexion / inversion,
- helpers de séries delta.

### `firmware/src/services/mfcc.cpp`

Le vrai pipeline audio vers features :

- buffers temporaires,
- traitement FFT,
- projection Mel,
- projection DCT,
- calcul des deltas,
- statistiques en continu,
- vecteur final de 194 valeurs.

### `firmware/src/main.cpp`

Initialisation runtime et boucle d’inférence :

- initialisation du `Serial`,
- chargement du modèle TFLite,
- allocation des tenseurs,
- capture audio,
- extraction des features,
- inférence,
- affichage du verdict.

## État de validation

Après le refactor, `inv build` réussit.

Cela signifie que le débordement du linker est résolu et que le firmware tient à nouveau dans la DRAM disponible.

## Point d’attention restant

La pression mémoire a été fortement réduite, mais le firmware utilise encore des allocations temporaires sur le heap pendant l’extraction des features.

C’est acceptable pour la cible actuelle, mais si la stabilité à l’exécution devient un problème plus tard, la prochaine optimisation serait de :

- réutiliser plus agressivement un seul buffer de travail,
- ou déplacer davantage du pipeline de features vers un modèle purement streaming.

Pour l’objectif de test actuel, le problème de build est corrigé.

## Résumé court

Le débordement DRAM a été résolu en :

- réduisant l’arène TFLite,
- supprimant les gros buffers MFCC globaux,
- déplaçant l’état d’extraction vers des allocations temporaires,
- utilisant des statistiques en continu au lieu de matrices stockées,
- en gardant uniquement de petits coefficients persistants dans `mfcc_utils.cpp`.
