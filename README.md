# 💼 Classification et Reclassification de Produits E-commerce

Ce projet permet de :
1. Compléter automatiquement les catégories manquantes (`Univers`, `Nature`) d’un produit e-commerce.
2. Détecter des anomalies dans les étiquettes existantes à l’aide d’embeddings CamemBERT.
3. Suggérer des reclassifications plus cohérentes.
4. Extraire automatiquement des attributs comme **les couleurs** et **les dimensions**.

---

## 📦 Installation

```bash
pip install pandas numpy scikit-learn nltk unidecode sentence-transformers
```

```python
import nltk
nltk.download('stopwords')
```

---

## 📁 Structure des fichiers

```text
.
├── preprocessing.py         # Nettoyage, extraction d'attributs
├── modeling.py              # Modèles de prédiction supervisée
├── camembert_reclass.py     # Embeddings + suspicion + suggestion
├── data/                    # Données CSV
├── embeddings/              # Sauvegarde des embeddings
└── README.md
```

---

## 🔧 1. Nettoyage et préparation des données

```python
from preprocessing import Processing, TextCleaner

p = Processing("data/produits.csv")
p.inspect_data()
p.remove_unused_columns()
p.remove_rows_with_all_labels_missing()
p.clean_labels()
df = p.get_data()

tc = TextCleaner(df)
tc.apply_cleaning()
df_clean = tc.get_data()
```

---

## 🧠 2. Prédiction des catégories manquantes

```python
from modeling import ProcessingModeler

modeler = ProcessingModeler(df_clean)
df_pred = modeler.predict_missing_univers()
```

---

## 🎨 3. Extraction de couleurs et dimensions

```python
from preprocessing import FeatureExtractor

extractor = FeatureExtractor(df_pred)
extractor.apply()
df_feat = extractor.get_data()
```

---

## 🧪 4. Détection de suspicion + suggestions CamemBERT

```python
from camembert_reclass import CamembertReclassifier

camembert = CamembertReclassifier()
df_final = camembert.run(df_feat)
```

---

## 💾 (Optionnel) Sauvegarde/Rechargement des embeddings

```python
# Sauvegarde
import pickle
with open("embeddings/df_embeddings.pkl", "wb") as f:
    pickle.dump(df_final, f)

# Chargement
camembert = CamembertReclassifier(precomputed_embeddings_path="embeddings/df_embeddings.pkl")
df_final = camembert.run(df_final)
```

---

## 📊 Colonnes finales produites

- `Univers`, `Nature` : Catégories initiales ou complétées
- `sim_univers`, `sim_nature` : Similarités Cosine avec le libellé
- `univers_suspect`, `nature_suspect` : Drapeau de suspicion
- `Univers suggéré`, `Nature suggérée` : Catégories recommandées si suspicion
- `couleurs`, `dimensions` : Attributs extraits

---

## 🔀 Pipeline résumé

```python
from preprocessing import Processing, TextCleaner, FeatureExtractor
from modeling import ProcessingModeler
from camembert_reclass import CamembertReclassifier

# 1. Prétraitement
df = Processing("data/produits.csv").remove_unused_columns().remove_rows_with_all_labels_missing().clean_labels().get_data()
df = TextCleaner(df).apply_cleaning().get_data()

# 2. Prédiction
df = ProcessingModeler(df).predict_missing_univers()

# 3. Extraction attributs
df = FeatureExtractor(df).apply().get_data()

# 4. Détection & suggestion
df_final = CamembertReclassifier().run(df)
```

---

## 📌 Paramètres importants

- `threshold_quantile` : Seuil de similarité basé sur les quantiles (ex: 0.25)
- `confidence_margin` : Marge exigée entre catégorie actuelle et catégorie suggérée (ex: 0.1)

---

## 🛠️ Développement

Réalisé par **Fatima Aboura** – Test technique pour **Nricher**, avril 2025.

Librairie utilisée pour les embeddings :  
[📚 SentenceTransformers - CamemBERT](https://huggingface.co/dangvantuan/sentence-camembert-large)
