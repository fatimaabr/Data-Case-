# Classification de Produits e-Commerce

Ce projet vise à enrichir un dataset e-commerce en complétant les catégories manquantes (`Univers`, `Nature`) et en proposant des reclassements pertinents via un pipeline combinant des méthodes supervisées, des règles, et des embeddings avec CamemBERT.

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

- Suppression des colonnes inutiles (`Cod_cmd`, `Vendeur`, etc.)
- Suppression des lignes avec `Libellé produit` manquant
- Nettoyage du texte (minuscule, accents, ponctuations)
- Suppression des doublons

### 2. Traitement des catégories

- Suppression des lignes sans `Univers` et `Nature`
- Prédiction des `Univers` manquants à partir de `Libellé produit` via un modèle TF-IDF + Logistic Regression

### 3. Feature Engineering

- Extraction des couleurs via correspondances regex
- Extraction des dimensions (simple : `40cm`, double : `120x200cm`)

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
│   └── e_commerce.csv
├── results/
│   └── resultat_reclasse.csv
└── README.md
```
