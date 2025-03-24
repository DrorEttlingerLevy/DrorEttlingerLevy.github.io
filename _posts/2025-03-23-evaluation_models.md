---
title: 'Evaluations Models'
date: 2025-03-23
permalink: /posts/2025/02/evaluations_models/
---
A major aspect of ML project is the evaluations part, I participeted in a IBM confrence where the speaker told about an intir experct team only dedicated to evaluate the LLM preformence and how to do this. for each type of probelm and model there are differnt evaluation metrices.

# Classification
when using grid search, then we calculate the predicted scores and use this comperisan between true and predicted to evaluet the model preformences.

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

Now for the evaluation:

precision_recall_curve works only for binary classification
For KNeighborsClassifier, it often memorizes training data, so it gives nearly perfect predictions on it.
label_binarize already does one-vs-rest
![alt text](precision-recall.png)

```python

y_test_pred = grid_search.best_estimator_.predict(X_test)

confusion_matrix = ic(confusion_matrix(y_test, y_test_pred))
ic(precision_score(y_test, y_test_pred, average='macro'))
ic(recall_score(y_test, y_test_pred, average='macro'))
ic(f1_score(y_test, y_test_pred, average='macro'))
```

ploting the confusion matrix error

```python
plt.matshow(confusion_matrix, cmap=plt.cm.gray)

row_sums = confusion_matrix.sum(axis=1, keepdims=True)
norm_conf = confusion_matrix/row_sums
np.fill_diagonal(norm_conf, 0)
plt.matshow(norm_conf,cmap=plt.cm.gray)
plt.show
```

for ploting
```python
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_test_scores = grid_search.best_estimator_.predict_proba(X_test)
prec, rec, thresh = precision_recall_curve(y_test_bin[:, 5], y_test_scores[:, 5])

# Remove last point to match threshold length
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

for i in range(10):  # for digits 0–9
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_test_scores[:, i])
    plt.plot(rec, prec, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid()
plt.show()


for i in range(10):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_test_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Digits")
plt.legend()
plt.grid()
plt.show()
```

