# KI-Methoden für den MNIST-Datensatz — Logistische Regression

**Klossouan Dangui Melchi** | Gruppe 205 | WS 2025/26  
Modul: Künstliche Intelligenz I | Betreuer: Prof. Dr. Christian Heiliger, Dr. Jan-Matthis Waack  
Studiengang: Data Science | Justus-Liebig-Universität Gießen

---

## Schwerpunkt

Dieses Repository enthält meine individuelle Implementierung der **Logistischen Regression (LR)** zur Klassifikation handgeschriebener Ziffern (0–9) im MNIST-Datensatz.

**Leitfrage:** Wie gut klassifiziert ein lineares Modell einen nichtlinearen Bildklassifikationsdatensatz, und welchen Einfluss haben Datenaugmentierung, Hyperparameter-Tuning und Dimensionsreduktion auf die Generalisierungsfähigkeit?

---

## Ergebnisse auf einen Blick

| Experiment | CV-Accuracy | Test-Accuracy |
|---|---|---|
| LR + Augmentierung + GridSearch (lbfgs, l2) | 85,35 % | 85,10 % |
| **LR ohne Augmentierung — Basismodell** | — | **92,64 %** |
| PCA (200 Komponenten) + LR | — | 92,64 % |
| Korrigierter GridSearch (saga, l2, C=0.1) | 92,12 % | 92,58 % |
| CNN (Referenz Vorlesung) | — | ≈ 98–99 % |

---

## Projektstruktur

```
logistic_regression_mnist/
├── notebook.ipynb          # Jupyter Notebook mit allen Experimenten
├── README.md               # Diese Datei
└── figures/
    ├── confusion_matrix.png
    ├── learning_curve.png
    └── weight_visualization.png
```

---

## Voraussetzungen

```bash
pip install numpy matplotlib scikit-learn tensorflow opencv-python
```

| Bibliothek | Version |
|---|---|
| Python | 3.11 |
| scikit-learn | 1.x |
| TensorFlow | 2.x |
| NumPy | — |
| Matplotlib | — |
| OpenCV | 4.x |

---

## Datensatz

MNIST wird automatisch via `tensorflow.keras.datasets.mnist` geladen:
- 60 000 Trainingsbilder, 10 000 Testbilder
- 28 × 28 Pixel, Klassen 0–9
- Normalisierung: Division durch 255 → Wertebereich [0, 1]
- Flattening: `(N, 28, 28)` → `(N, 784)`

---

## Experimente

### Experiment 1 — LR mit Datenaugmentierung + GridSearch

Trainingsdaten auf 180 000 Bilder verdreifacht durch:
- Rotation (10° via `cv.warpAffine`)
- Gaußsches Rauschen (σ = 0,1)

GridSearchCV (5-fold, `C ∈ {0.01, 0.1, 1, 10}`, `penalty=l2`, `solver=lbfgs`)  
→ Bestes Ergebnis: `C=0.01`, CV-Acc = 85,35 %, Test-Acc = **85,10 %**

> ⚠️ Problem entdeckt: Inkonsistenz zwischen `penalty=None` im Basismodell und `penalty=l2` im Suchraum. In Experiment 3 behoben.

---

### Experiment 2 — LR ohne Augmentierung (Basismodell)

```python
LogisticRegression(max_iter=500, random_state=42)
# Standardparameter: penalty=l2, solver=lbfgs, C=1.0
```

Direkt auf den originalen 60 000 Bildern trainiert.  
→ Test-Accuracy = **92,64 %** (+7,54 PP gegenüber Experiment 1)

**Erkenntnis:** Die LR lernt keine geometrische Invarianz. Rotierte Bilder erzeugen widersprüchliche Pixelsignale für dieselbe Klasse — Augmentierung schadet linearen Modellen.

---

### Experiment 3 — PCA-Pipeline + korrigierter GridSearch

**PCA+LR-Pipeline** (`sklearn.pipeline.Pipeline`, `whiten=True`):

| n_components | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|
| Test-Accuracy | 91,18 % | 92,21 % | 92,36 % | **92,64 %** | 92,41 % |

Maximum bei N=200 (784 → 200 Merkmale), danach sinkend durch rauschbehaftete Komponenten.

**Korrigierter GridSearch** (`solver=saga`, `C ∈ {0.001, 0.01, 0.1, 1, 10}`, `penalty ∈ {l1, l2}`):  
→ Bestes Modell: `C=0.1, l2`, CV-Acc = 92,12 %, Test-Acc = 92,58 %

> L1 mit `C=0.001` kollabierte auf 69,93 % — nahezu alle Gewichte wurden auf null regularisiert.

---

## Analyse

### Confusion Matrix

736 Fehler von 10 000 Testbildern (7,36 % Fehlerrate).  
Stärkste Verwechslungspaare: **5 ↔ 3**, **9 ↔ 4**, **8 ↔ 3**  
Schlechteste Klassen: 5 und 8 (je F1 = 0,88) — ähnliche Kurvenform.  
Beste Klasse: 1 (F1 = 0,97) — kaum Verwechslungspartner.

### Lernkurven-Analyse

Klassisches **Underfitting-Muster (hoher Bias)**:
- Train-Accuracy: ≈ 98,7 % (4 800 Bilder) → ≈ 94 % (48 000 Bilder)
- Validation-Accuracy: flacht bei ≈ 91,9 % ab, konvergiert nicht gegen Train

Mehr Daten allein können die strukturelle Schwäche des linearen Modells nicht beheben.

### Gewichtsvisualisierung

`lr.coef_[k].reshape(28, 28)` visualisiert für alle 10 Klassen:
- **Rot** = Pixel, die für die Klasse sprechen
- **Blau** = Pixel, die gegen die Klasse sprechen

Hauptvorteil der LR gegenüber neuronalen Netzen: vollständige Interpretierbarkeit.

---

## Fazit

Das endgültige Modell ist:

```python
LogisticRegression(max_iter=500, random_state=42)
# Test-Accuracy: 92.64%
```

Alle Varianten konvergieren bei ≈ 92,6 % — die LR ist strukturell nicht in der Lage,
die nichtlinearen Entscheidungsgrenzen im Pixelraum abzubilden (CNN: > 99 %).  
Als nächster Schritt bietet sich ein `MLPClassifier` mit einer versteckten Schicht an.

---

## Einsatz generativer KI

Claude (Anthropic) wurde ausschließlich für Debugging-Unterstützung und LaTeX-Formatierung eingesetzt — nicht zur Generierung von Ergebnissen oder Code.

Beispielhafte Prompts:
- *„Mein GridSearchCV gibt eine ConvergenceWarning aus bei max_iter=200. Wie behebe ich das?"*
- *„Erkläre mir die Inkonsistenz zwischen penalty=None im Basismodell und penalty=l2 im GridSearch-Suchraum."*

Alle inhaltlichen Entscheidungen wurden eigenständig erarbeitet.

---

## Quellen

- LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST database. http://yann.lecun.com/exdb/mnist/
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
- Heiliger, C. & Waack, J.-M. (2025/26). Vorlesungsfolien: Künstliche Intelligenz I. JLU Gießen.
