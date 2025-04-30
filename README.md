# Classification de Produits e-Commerce

Ce projet vise à enrichir un dataset e-commerce en complétant les catégories manquantes (`Univers`) et en proposant des reclassements pertinents via un pipeline combinant des méthodes supervisées, des règles, et des embeddings avec CamemBERT.

---

## Sommaire

1. [Pré-requis](#pré-requis)
2. [Pipeline général](#pipeline-général)
3. [Étapes détaillées](#étapes-détaillées)
4. [Embedding & Reclassement avec CamemBERT](#embedding--reclassement-avec-camembert)
5. [Exécution](#exécution)
6. [Résultats attendus](#résultats-attendus)
7. [Structure du dépôt](#structure-du-dépôt)

---

## Pré-requis

- Python ≥ 3.8
- `scikit-learn`, `pandas`, `nltk`, `sentence-transformers`, `unidecode`

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Pipeline général

1. Chargement et inspection des données
2. Nettoyage des libellés produits
3. Traitement des colonnes manquantes (Univers, Nature)
4. Extraction de features (couleurs, dimensions)
5. Embedding via CamemBERT et suggestion de reclassement

---

## Étapes détaillées

### 1. Chargement & Nettoyage

- Suppression des colonnes inutiles (`Cod_cmd`, `Vendeur`,`Date de commande`,	`Montant cmd`, `Quantité`,	`Prix transport`,	`Délai transport annoncé`)
- Suppression des lignes avec `Libellé produit` manquant
- Nettoyage du texte  `Libellé produit` (minuscule, accents, ponctuations)
- Suppression des doublons (`Libellé produit`) (Réduit significativement la taille du dataset)

### 2. Traitement des catégories

- Suppression des lignes sans `Univers` et `Nature` (Sont généralement des services et non pas des produits)
- Prédiction des `Univers` manquants à partir de `Libellé produit` via un modèle TF-IDF + Logistic Regression

### 3. Feature Engineering

- Extraction des couleurs via correspondances regex
- Extraction des dimensions :
  - Les dimensions simples comme `40cm` sont détectées via des expressions régulières ciblant des motifs numériques suivis de `cm`, `mm`, etc.
  - Les dimensions doubles comme `120x200cm` sont identifiées avec des motifs de type `nombre x nombre` suivis d'une unité.
  - Une normalisation est appliquée pour unifier les formats extraits.

### 4. Embedding & Reclassement avec CamemBERT

L’objectif est de vérifier si les étiquettes `Univers` et `Nature` déjà présentes dans les données sont cohérentes avec le contenu du `Libellé produit`.

#### Méthode :

1. **Embedding** : chaque `Libellé produit` est transformé en vecteur avec `SentenceTransformer("dangvantuan/sentence-camembert-large")`.
2. **Vecteurs de référence** : pour chaque catégorie (`Univers`, `Nature`), on calcule un vecteur moyen à partir des produits connus de cette catégorie.
3. **Similarité cosinus** : on compare le vecteur du produit à celui de sa catégorie :
   - Si la similarité est **faible** avec la catégorie assignée
   - Et qu'une autre catégorie a une **similarité plus élevée**
   → Cela est considéré comme **potentiellement anormal** : une suggestion de reclassement est alors faite.

💡 Cette approche permet de **détecter des incohérences sémantiques** entre le texte du produit et sa classification actuelle.

💡 **Note :** Pour des raisons de limitations CPU/GPU, les embeddings ont été générés sur un sous-échantillon de 10 000 lignes :
```python
# df = df_full.sample(n=10000, random_state=42).reset_index(drop=True)
```

---

## Exécution

L’ensemble des étapes est contenu dans un **notebook Colab unique** incluant toutes les classes et l'exécution complète du pipeline.

---

## Résultats attendus

Le fichier final contient :

- `Univers` et `Nature` remplis ou reclassés
- Suggestions de reclassement (`Univers suggéré`, `Nature suggérée`)
- Scores de similarité
- Couleurs et dimensions détectées

---

## Structure du dépôt

```bash
.
├── classification_pipeline.ipynb   # Notebook complet avec tout le code (classes et pipeline)
├── requirements.txt
├── data/
│   └── dataset_nettoye.csv
├── results/
│   └── resultat_reclasse.csv
└── README.md
```

## Classes principales

- `Processing`
  - Inspection des données (dimensions, types, premières et dernières lignes).
  - Vérification des valeurs manquantes et des doublons.
  - Suppression des colonnes inutiles et des lignes avec des étiquettes manquantes.

- `TextCleaner`
  - Nettoie les libellés produits (minuscule, accents, ponctuation, etc.)
  - Méthodes principales : `clean_text()`, `remove_punctuation()`, `normalize_text()`

- `ProcessingModeler`
  - Prédit les catégories manquantes (`Univers`) via un modèle TF-IDF + LogisticRegression
  - Méthodes : `fit()`, `predict()`

- `FeatureExtractor`
  - Extrait des attributs produits comme : couleurs, dimensions simples, dimensions doubles
  - Méthodes : `extract_colors()`, `extract_dimensions()`

- `CamembertReclassifier`
  - Encode les libellés produits avec CamemBERT
  - Calcule les similarités cosinus pour évaluer la cohérence des catégories
  - Méthodes : `compute_embeddings()`, `compute_centroids()`, `suggest_reclassification()`

---
## 📊 Visualisation des reclassements majeurs
Le graphique ci-dessous montre les 10 univers originaux ayant connu le plus de reclassements, ainsi que les nouvelles catégories proposées :

![Changements d'univers](téléchargement%20resss.png)

## 📈 Analyse des résultats et justifications des erreurs
Suite à l'application du modèle de recatégorisation, plusieurs reclassements erronés ou surprenants ont été observés. Bien que certaines suggestions soient cohérentes, d'autres révèlent les limites du modèle. Voici les principales raisons possibles :

### 1. Seuil de similarité trop permissif
Le modèle accepte une reclassification si la similarité cosinus dépasse un certain seuil. Un seuil trop bas (ex. 0.6) peut autoriser des suggestions peu fiables.

➡️ *Exemple* : une similarité de 0.63 peut sembler proche mais indique une confiance moyenne.

### 2. Données d'entraînement incomplètes ou déséquilibrées
Certaines catégories sont sous-représentées dans l'ensemble d'apprentissage, biaisant l'estimation des vecteurs moyens (centroïdes).

➡️ *Exemple* : une catégorie comme "Bureau Rangement" avec peu d'exemples sera mal modélisée.

### 3. Libellés produits peu explicites
Le modèle ne dispose que du champ "Libellé produit", souvent trop court ou ambigu.

➡️ *Exemple* : un libellé comme "Ensemble 3 pièces" n’indique rien de spécifique.

### 4. Proximité sémantique entre catégories
Certaines catégories sont sémantiquement proches (ex. "Chambre Literie" vs "Décoration Textile"), rendant leur distinction difficile sans connaissances métiers supplémentaires.

## ✅ Recommandations
- Élever le seuil de similarité pour éviter les reclassifications peu fiables.
- Ajouter des données contextuelles si disponibles (ex. description, usage, attributs).
- Introduire une étape de validation humaine pour les suggestions incertaines.
- Affiner les centroïdes en rééquilibrant les catégories sous-représentées.

---


## Limites et perspectives

- Le modèle TF-IDF + LogisticRegression reste sensible aux formulations non standards.
- L’approche par embeddings dépend fortement de la qualité du sous-échantillon.

### Améliorations possibles :

- Intégration de modèles de classification plus robustes (CamemBERT finetuné, LLM)
- Reclassement automatique avec seuils adaptatifs plutôt que suggestion simple
- Interface utilisateur pour la revue manuelle des cas ambigus
- Évaluation systématique des performances (classification accuracy, F1-score)
