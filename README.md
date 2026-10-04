# Prédiction de la survie des passagers du Titanic

Projet de classification binaire réalisé avec Python et Scikit-learn pour prédire la survie des passagers du Titanic.

Le projet couvre une démarche complète de machine learning :

- analyse exploratoire ;
- traitement des données manquantes ;
- feature engineering ;
- création d'un pipeline reproductible ;
- comparaison de plusieurs modèles ;
- optimisation des hyperparamètres ;
- évaluation sur un jeu de validation indépendant ;
- analyse des erreurs et interprétation ;
- génération d'une soumission Kaggle.

## Problématique

L'objectif est de prédire la variable `Survived` à partir des caractéristiques disponibles pour chaque passager :

- `0` : le passager n'a pas survécu ;
- `1` : le passager a survécu.

Il s'agit d'un problème de classification binaire.

Le jeu d'entraînement contient 891 passagers étiquetés. Le jeu de test contient 418 passagers pour lesquels le pipeline final génère les prédictions destinées à Kaggle.

## Résultats

Le modèle sélectionné est une régression logistique intégrée dans un pipeline complet de préparation des données.

| Métrique | Résultat |
|---|---:|
| Accuracy moyenne en validation croisée | 0,829 |
| Accuracy sur le jeu de validation | 0,821 |
| Précision sur le jeu de validation | 0,768 |
| Rappel sur le jeu de validation | 0,768 |
| F1-score sur le jeu de validation | 0,768 |
| ROC-AUC sur le jeu de validation | 0,874 |

La baseline, qui prédit systématiquement la classe majoritaire, obtient une accuracy d'environ 61 %. Le modèle final améliore donc nettement cette référence.

La proximité entre l'accuracy moyenne en validation croisée et celle obtenue sur le jeu de validation suggère une généralisation cohérente, sans signe manifeste de surapprentissage.

> Le score Kaggle n'est pas indiqué tant que la soumission n'a pas été évaluée sur le leaderboard. Les résultats ci-dessus proviennent du protocole d'évaluation local.

## Analyse exploratoire

L'analyse exploratoire a porté sur :

- la distribution de la variable cible ;
- les valeurs manquantes ;
- le taux de survie selon le sexe ;
- le taux de survie selon la classe du billet ;
- l'interaction entre le sexe et la classe ;
- la distribution de l'âge ;
- le tarif du billet ;
- le port d'embarquement ;
- la composition familiale ;
- les titres extraits du nom ;
- les informations relatives à la cabine.

### Principaux constats

- Les femmes présentent un taux de survie nettement supérieur à celui des hommes.
- Les passagers de première classe ont davantage survécu que ceux de troisième classe.
- Le sexe et la classe sont les variables les plus fortement associées à la survie.
- Les enfants semblent avoir bénéficié d'un taux de survie supérieur à plusieurs groupes adultes.
- Les passagers voyageant seuls présentent généralement un taux de survie inférieur à celui des passagers accompagnés.
- Les tarifs élevés sont davantage représentés parmi les survivants, mais cette relation est en partie liée à la classe du billet.

Ces résultats sont descriptifs et ne doivent pas être interprétés comme des relations causales.

## Qualité des données

Les données brutes comportent plusieurs valeurs manquantes :

- `Age` contient des âges non renseignés ;
- `Cabin` possède une proportion importante de valeurs manquantes ;
- `Embarked` contient quelques valeurs absentes dans le jeu d'entraînement ;
- `Fare` contient une valeur manquante dans le jeu de test.

Les imputations ne sont pas appliquées directement aux fichiers : elles sont intégrées au pipeline Scikit-learn afin d'être ajustées uniquement sur les données d'entraînement.

## Feature engineering

| Variable | Description |
|---|---|
| `FamilySize` | Nombre total de membres de la famille à bord, passager inclus |
| `IsAlone` | Indique si le passager voyage seul |
| `FamilyGroup` | Regroupement des passagers selon la taille de leur famille |
| `Title` | Titre extrait du nom : Mr, Mrs, Miss, Master ou Rare |
| `CabinKnown` | Indique si l'information sur la cabine est disponible |
| `Deck` | Pont extrait du numéro de cabine |
| `TicketPrefix` | Préfixe non numérique du billet |
| `NameLength` | Nombre de caractères du nom complet |

Les transformations sont regroupées dans `src/features.py`. La fonction de feature engineering :

- n'utilise pas la variable cible ;
- ne modifie pas les données brutes ;
- n'apprend aucun paramètre statistique ;
- est appliquée de façon identique aux données d'entraînement, de validation et de test.

## Pipeline de machine learning

Le pipeline final applique successivement :

1. la création des variables ;
2. la sélection des caractéristiques ;
3. l'imputation des variables numériques par la médiane ;
4. la standardisation des variables numériques ;
5. l'imputation des variables catégorielles par la modalité la plus fréquente ;
6. le one-hot encoding des variables catégorielles ;
7. la classification par régression logistique.

`Pipeline` et `ColumnTransformer` évitent les fuites de données et garantissent l'application des mêmes transformations à l'entraînement et à la prédiction.

## Modèles comparés

- `DummyClassifier` ;
- régression logistique ;
- arbre de décision ;
- forêt aléatoire ;
- gradient boosting.

Les modèles ont été comparés par validation croisée stratifiée à cinq plis sur le jeu de développement. La régression logistique a été retenue, puis ses hyperparamètres ont été optimisés avec les outils de recherche de Scikit-learn.

## Protocole d'évaluation

1. Séparation de 20 % des observations dans un jeu de validation stratifié.
2. Comparaison des modèles uniquement sur les 80 % restants.
3. Validation croisée stratifiée à cinq plis.
4. Optimisation des hyperparamètres du meilleur candidat.
5. Évaluation unique du pipeline sélectionné sur le jeu de validation.
6. Réentraînement du pipeline final sur les 891 observations.
7. Génération des prédictions pour le jeu de test Kaggle.

Métriques utilisées : accuracy, précision, rappel, F1-score, ROC-AUC et matrice de confusion.

L'accuracy est la métrique principale, car c'est celle de la compétition Kaggle. Les autres métriques permettent d'étudier plus finement le comportement du modèle sur la classe des survivants.

## Analyse des erreurs

Les prédictions du jeu de validation ont été analysées pour identifier :

- les faux positifs : passagers prédits survivants alors qu'ils n'ont pas survécu ;
- les faux négatifs : passagers prédits non-survivants alors qu'ils ont survécu ;
- les performances selon le sexe ;
- les performances selon la classe ;
- les performances selon la combinaison sexe × classe.

Une importance par permutation a également été calculée pour mesurer l'utilité prédictive des variables brutes. Ces résultats décrivent le comportement du modèle et ne représentent pas des relations causales.

## Organisation du projet

```text
titanic-ml-project/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   └── features.py
│
├── models/
│   └── titanic_pipeline.joblib
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── hyperparameter_search_results.csv
│   ├── validation_metrics.json
│   ├── classification_report.csv
│   ├── permutation_importance.csv
│   └── validation_error_analysis.csv
│
├── submissions/
│   ├── submission.csv
│   └── test_probabilities.csv
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

Certains fichiers générés (modèle, données) peuvent être exclus du dépôt via `.gitignore`.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Ingeniir/titanic-project.git
cd titanic-project
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Activation :

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger les données

Depuis la [compétition Kaggle Titanic](https://www.kaggle.com/competitions/titanic), télécharger `train.csv`, `test.csv` et `gender_submission.csv`, puis les placer dans :

```text
data/raw/
```

Les données ne sont pas nécessairement incluses dans le dépôt.

### 5. Lancer Jupyter

```bash
jupyter notebook
```

Exécuter les notebooks dans l'ordre :

1. `01_data_exploration.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`

## Livrables générés

Après l'exécution du notebook de modélisation :

- `reports/model_comparison.csv`
- `reports/hyperparameter_search_results.csv`
- `reports/validation_metrics.json`
- `reports/classification_report.csv`
- `reports/permutation_importance.csv`
- `reports/validation_error_analysis.csv`
- `models/titanic_pipeline.joblib`
- `submissions/submission.csv`
- `submissions/test_probabilities.csv`

Le fichier à envoyer sur Kaggle est `submissions/submission.csv`.

## Technologies utilisées

Python · pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter · joblib

## Compétences mises en œuvre

- analyse exploratoire et visualisation de données ;
- traitement des valeurs manquantes ;
- feature engineering ;
- pipelines Scikit-learn ;
- validation croisée stratifiée ;
- optimisation d'hyperparamètres ;
- comparaison de modèles et évaluation d'un classifieur ;
- analyse des erreurs ;
- interprétabilité par permutation ;
- sérialisation d'un modèle ;
- génération d'une soumission Kaggle.

## Limites

- Le jeu d'entraînement ne contient que 891 observations.
- Titanic est un dataset pédagogique très étudié.
- Les performances peuvent varier selon le découpage des données.
- Certaines modalités contiennent peu d'observations.
- Plusieurs passagers appartiennent aux mêmes familles ou partagent le même billet, ce qui crée des dépendances entre observations.
- Une importance prédictive ne démontre pas une relation causale.
- Le jeu de validation reste relativement petit.

## Améliorations possibles

- comparer la validation stratifiée à une validation groupée par famille ou billet ;
- mesurer l'apport individuel de chaque variable créée ;
- tester une imputation de l'âge conditionnée par le titre et la classe ;
- étudier la calibration des probabilités ;
- simplifier les variables redondantes ;
- automatiser l'entraînement avec des scripts en ligne de commande ;
- déployer le pipeline dans une application Streamlit.

## Conclusion

Ce projet montre qu'une démarche de machine learning ne se limite pas à entraîner un algorithme : la préparation des données, la prévention des fuites, la validation et l'analyse des erreurs sont essentielles pour obtenir des résultats interprétables et reproductibles.

La régression logistique sélectionnée atteint une accuracy moyenne de 82,9 % en validation croisée et de 82,1 % sur le jeu de validation, avec une ROC-AUC de 0,874.