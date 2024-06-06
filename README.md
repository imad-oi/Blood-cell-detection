# Projet:Détection Automatisée des Cellules Cancéreuses dans le Sang

Ce projet a pour objectif de développer un système automatisé permettant la détection
et la caractérisation des cellules cancéreuses dans les échantillons sanguins, afin de déter-
miner le stade du cancer. Cette initiative repose sur l’utilisation de techniques avancées
de traitement d’images et d’apprentissage automatique. Elle revêt une importance cru-
ciale pour un diagnostic précoce du cancer, notamment la leucémie lymphoblastique aiguë
(ALL), afin d’améliorer les chances de traitement et les résultats cliniques pour les pa-
tients

![Web App](web-app.png)

# Structure de projet
```bash
├───data
│   ├───segmented_test
│   │   ├───benign
│   │   ├───EarlyPreB
│   │   ├───PreB
│   │   └───ProB
│   ├───segmented_train
│   │   ├───benign
│   │   ├───EarlyPreB
│   │   ├───PreB
│   │   └───ProB
│   ├───test_data
│   │   ├───benign
│   │   ├───EarlyPreB
│   │   ├───PreB
│   │   └───ProB
│   └───train_data
│       ├───benign
│       ├───EarlyPreB
│       ├───PreB
│       └───ProB
├───models
├───flask
├───notebooks
└───src
```

## Dossier Data
Ce répertoire contient le jeu de données utilisé dans ce projet. Il comprend à la fois les images originales et les images segmentées, à la fois pour les données d'entraînement et de test.

## Models
Ce répertoire contient les modèles entraînés.

## Flask
Ce répertoire contient une application web avec flask.

## notebooks 
Ce répertoire contient le notebook détaillant toutes les étapes et les processus effectués pour réaliser ce projet.

## Src
Ce répertoire contient deux fichiers :
1. inference.py : Ce fichier est utilisé pour exécuter un serveur web permettant d'utiliser les modèles. Pour lancer le serveur, exécutez la commande suivante à l'intérieur du dossier `src` : `streamlit run .\inference.py`.
2. features.py : Ce fichier contient quelques fonctions utilitaires utilisées pour préparer les images pour la classification.

# Résultats
| modèle | KNN       | SVM       | Arbre de decision | Foret Aleatoire |
|-----------|-----------|-----------|-------------------|-----------------|
| Précision (%) |    92.6   |     93.1   |      88.9         |     92.9        |"# Blood-cell-detection" 
