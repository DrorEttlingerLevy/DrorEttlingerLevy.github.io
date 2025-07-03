---
title: 'Evaluations Models'
date: 2025-03-23
permalink: /posts/2025/02/evaluations_models/
image: /images/blog/03072025_ibm.png
preview: >
  A major aspect of any ML project is the evaluation part. I participated in an IBM conference where the speaker described an entire expert team dedicated solely to evaluating LLM performance and how to approach it. For each type of problem and model, there are different evaluation metrics.
header:
  teaser: /blog/03072025_ibm.png
---
A major aspect of any machine learning project is the evaluation part.  
I was in an IBM conference where one speaker described an entire expert team that only works on evaluating LLMs.  
It made me realize how serious and non-trivial this stage is.

Each type of problem like classification, regression, multi-label, comes with its own **evaluation metrics**.

---

## 🔢 Classification with Grid Search

Let’s look at one typical classification setup with `KNeighborsClassifier` and `GridSearchCV`.

```python
pipeline = Pipeline(
    [("preprocessing", StandardScaler()), ("model", KNeighborsClassifier())]
)

param_grid = {
    "model__n_neighbors": [2, 3, 5, 7],
    "model__weights": ["uniform", "distance"],
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

scores_grid = ic(grid_search.cv_results_)
ic(grid_search.best_estimator_)
```

<div class="note">
  <strong>Note:</strong> KNN tends to memorize the training data, so don't be surprised by near-perfect training accuracy.
</div>

---

## ✅ Basic Evaluation on Test Set

```python
y_test_pred = grid_search.best_estimator_.predict(X_test)

confusion_matrix = ic(confusion_matrix(y_test, y_test_pred))
ic(precision_score(y_test, y_test_pred, average='macro'))
ic(recall_score(y_test, y_test_pred, average='macro'))
ic(f1_score(y_test, y_test_pred, average='macro'))
```

<div class="info">
  <strong>Reminder:</strong> Use <code>average='macro'</code> when doing multi-class classification to treat all classes equally.
</div>

---

## 🎛️ Confusion Matrix — Visualizing the Mistakes

```python
plt.matshow(confusion_matrix, cmap=plt.cm.gray)

row_sums = confusion_matrix.sum(axis=1, keepdims=True)
norm_conf = confusion_matrix / row_sums
np.fill_diagonal(norm_conf, 0)
plt.matshow(norm_conf, cmap=plt.cm.gray)
plt.show()
```

This shows not just the correct predictions, but the type and amount of wrong ones — important for debugging real models.

---

## 🔍 Precision-Recall Curve for Multi-class

```python
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_test_scores = grid_search.best_estimator_.predict_proba(X_test)

prec, rec, thresh = precision_recall_curve(y_test_bin[:, 5], y_test_scores[:, 5])
prec = prec[:-1]
rec = rec[:-1]

plt.plot(thresh, prec, label="Precision")
plt.plot(thresh, rec, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid()
plt.show()
```

<div class="note">
  <strong>Note:</strong> <code>precision_recall_curve</code> works only for binary problems — we apply it per class using one-vs-rest (via <code>label_binarize</code>).
</div>

Now plotting it for all classes:

```python
for i in range(10):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_test_scores[:, i])
    plt.plot(rec, prec, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid()
plt.show()
```

---

## 📈 ROC Curve — One per Class

```python
for i in range(10):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_test_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Digits")
plt.legend()
plt.grid()
plt.show()
```

<div class="info">
  <strong>Tip:</strong> The area under the ROC curve (AUC) gives a good overall measure — higher is better, and 1.0 is perfect.
</div>

---

This is a quick but full workflow for evaluating classification models, with good plots and insights to understand what’s going right and what’s not.