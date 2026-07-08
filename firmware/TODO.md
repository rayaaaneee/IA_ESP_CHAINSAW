# Un petit conseil supplémentaire pour l'ESP32 (C++) :
Si jamais tu prévois d'inclure ce fichier model.h dans plusieurs de tes fichiers C++ (par exemple dans main.cpp ET dans un audio_processor.cpp), le fait de définir un tableau directement dans un .h causera quand même une erreur au moment du "linkage".
Si c'est le cas, la meilleure pratique est de séparer en deux :

Un model.h qui contient juste : extern const unsigned char g_model_data[];

Un model.cpp qui contient le vrai tableau généré par ton script Python.

Mais si tu n'inclus model.h qu'une seule fois dans tout ton projet (ce qui est souvent le cas pour les petits projets TinyML), ton script avec la correction du #endif marchera parfaitement