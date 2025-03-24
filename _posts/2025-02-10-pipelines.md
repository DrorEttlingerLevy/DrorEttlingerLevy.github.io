---
title: 'Pipelines for DS Projects'
date: 2025-02-10
permalink: /posts/2025/02/pipelines/
---

Learning how to build **efficient pipelines** for data preprocessing and modeling in **Scikit-Learn** can streamline machine learning workflows.
This includes handling missing values, scaling, and encoding both numerical and categorical features.
The process involves using `Pipeline`, `ColumnTransformer`, and even custom transformers for feature engineering ⚙️
With `make_pipeline`, preprocessing and model training are combined seamlessly, optimizing performance using **`n_jobs=-1`** and **`remainder="passthrough"`**. Finally, saving and loading pipelines ensures reusability for future projects 💾🚀


So, we did the ordinary first steps:
1. load data
2. head, info, value_counts, describe, hist
3. test set 
4. make sure the test is representative --> StratifiedShuffleSplit
5. visualization of the features
6. correlation between features

Now, we need to understand how to create effective pipelines for preprocessing the data.

## Source: Hands On Machine Learning Book 📚
we have a few things we need to take care of:
1. missing data 🤷‍♀️
2. normalize data (0-1)✅
3. categorized data to turn to numeric📁

We'll set 2 diffract pipelines - for **numerical data and for categorical** data:

```python
# Define Pipeline for numeric
num_pipline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                        ])

# Define Pipeline for categoric
cat_pipline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
```

We identify the numeric and categorical columns

```python
train_set_cat = ['ocean_proximity']
train_set_num = list(train_set.drop(columns=['ocean_proximity']).columns)
```

Use preprocessor ColumnTransformet

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
  ('num', num_pipeline, train_set_num),
  ('cat', cat_pipeline, train_set_cat)
])
```

And finally fit_transform

```python
transform_train_set = preprocessor.fit_transform(train_set)
```

![My helpful screenshot](/images/070325.jpg)

```python
top_10_important_features = list(coefs.abs().sort_values(by='Coefficients', ascending=False).iloc[:10].index)
```
Now we'll integrate the feature selection into the pipeline 

```python 
from sklearn.feature_selection import SelectFromModel

ridgeCV=RidgeCV()
ridgecv_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectFromModel(ridgeCV, prefit=False, threshold='median')),
    ("model", ridgeCV)
])
ridgecv_pipeline.fit(X_train, y_train)
y_pred_ridgecv = ridgecv_pipeline.predict(X_test)

# to get the selected features names
feature_names = ridgecv_pipeline.named_steps["preprocessor"].get_feature_names_out()
# Get the feature selection step
feature_selector = ridgecv_pipeline.named_steps["feature_selection"]

# Get the mask of selected features (True for selected features, False for non-selected)
selected_features_mask = feature_selector.get_support()

# Get the names of selected features
selected_feature_names = feature_names[selected_features_mask]
```

![My helpful screenshot](/images/WhatsApp_Image_2025-02-10.jpg)


## Pipeline for Selecting Features [inria.github.io](https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html)

In linear models, the target value is modeled as a linear combination of the features.
Coefficients represent the relationship between the given feature $Xi$
 and the target $yi$
, assuming that all the other features remain constant (conditional dependence). This is different from plotting $Xi$
 versus $yi$
 and fitting a linear relationship: in that case all possible values of the other features are taken into account in the estimation (marginal dependence).

```python
# Extract trained RidgeCV model
ridge_model = ridgecv_pipeline.named_steps["model"]
# Get the feature names after preprocessing
feature_names = ridgecv_pipeline.named_steps["preprocessor"].get_feature_names_out()
# Get coefficients and ensure they match feature names
coefs = pd.DataFrame(
    ridge_model.coef_.reshape(-1, 1), columns=["Coefficients"], index=feature_names
)

# Plot
coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model Coefficients")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()
```

* The scale is from -1500 to 2000 because it matches the target feature (price), but it is already scaled in the pipeline so the importance of the features stays the same.

and a brief from the book Hands On Machine Learning:
![My helpful screenshot](/images/1739196533941.jpg)


We can also define our own **custom Transformer** if for example we want to add a new column 
```python
class OutletTypeEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Doesn't learn anything, just returns itself

    def transform(self, X):
        X_ = X.copy()  # Avoid modifying the original DataFrame
        X_['is_supermarket'] = X_['Outlet_Type'].isin([
            'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'
        ])
        return X_
```


where:
- `BaseEstimator`: Ensures the transformer follows Scikit-Learn’s API (allows hyperparameter tuning, cloning, etc.).
- `TransformerMixin`: Provides `fit_transform()` automatically if `fit()` and `transform()` are defined.

<div class="note">
    <strong>Note:</strong> It's is better and more flexible to use Pipelines and not make_pipeline (as shown below)
</div>

*`fit()`* --> learn the parameters needed to transform, for example the std and mean of a feature.
*`transform()`* --> actually transform the data (the train data), uses the std and the mean calculated before, on all the points of the feature.
*`fit_transfrom()`* --> do both fit (learning) and transform at the same time

<div class="note">
    <strong>Note:</strong> we'll *NOT* fit the test, because it will learn it's 
</div>


## Source: Youtube - Building a Machine Learning Pipeline with Python and Scikit-Learn | Step-by-Step Tutorial 🖥️

[Building a Machine Learning Pipeline with Python and Scikit-Learn \| Step-by-Step Tutorial - YouTube](https://www.youtube.com/watch?v=T9ETsSD1I0w&t=20s&ab_channel=Ryan%26MattDataScience)

a bit different way, first define the models, imputers and so on and then use `make_pipeline`

```python
imputer = SimpleImuter(strategy='mean')
lr = LogisticRegression()

pipeline1 = make_pipeline(imputer, lr)
pipeline1.fit(X_train,y_train)

pipeline1.score(X_train, y_train)
pipeline1.score(X_test, y_test)
```
![My helpful screenshot](/images/Pasted_image_20250210163007.jpg)


Also important to use the reminder and n_jobs for the ColumnTransformer
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ],
    remainder="passthrough",  # Keep unprocessed columns
    n_jobs=-1  # Use all available CPU cores for parallel processing
)
```

all left is to use make_pipeline to both the pipeline preprocessing and the model, and to fit the train data to it

![My helpful screenshot](/images/Pasted_image_20250210164222.jpg)


To the moment we all waited for, the score 🥳

```python
pipefinal.score(X_test, y_test)
```

How to save the pipeline

```python
import joblib
joblib.dump(pipefinal, 'pipe.joblib')

#to load
pipeline = joblib.load('pipe.joblib')
```

## Source: Medium Article
[Scikit-learn Pipelines Explained: Streamline and Optimize Your Machine Learning Processes \| by Sahin Ahmed, Data Scientist \| Medium](https://medium.com/@sahin.samia/scikit-learn-pipelines-explained-streamline-and-optimize-your-machine-learning-processes-f17b1beb86a4)

<div class="tip">
    <strong>tpp:</strong> other option for ColumnTransformer is `FeatureUnion` which works similarly but is more flexible, allowing any number of pipelines to be combined, not just column-wise operations.
</div>

![My helpful screenshot](/images/1739200574343.jpg)

We can also add to the pipeline a grid search CV for the preprocessing itself

```python
pipeline = Pipeline([
    ('preprocessing', StandardScaler()),  # This could be swapped with other scalers
    ('model', RandomForestRegressor())
])

param_grid = {
    'preprocessing': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'model__n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
ic(grid_search.cv_results)
```
To conclude all, a full titanic project using pipelines:

```python
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize,OneHotEncoder
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

#import the train data
df = pd.read_csv(r'titanic\train.csv')

y = df['Survived']
X = df.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


cat = X_train.select_dtypes(include="object").columns.tolist()
num = list(X_train.drop(columns=cat).columns)

drop_col = ['Cabin', 'Ticket']


num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler',StandardScaler())
                    ])

cat_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
  ('drop_col', 'drop', drop_col),
  ('num', num_pipeline, num),
  ('cat', cat_pipeline, cat)
])

model_pipeline = Pipeline([
  ('preprocessor', preprocessor),
  ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 5],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

for mean, std, params in zip(
    grid_search.cv_results_["mean_test_score"],
    grid_search.cv_results_["std_test_score"],
    grid_search.cv_results_["params"]
):
    print(f"{mean:.3f} ± {std:.3f} -> {params}")

# we will use the best model

y_test_pred = grid_search.best_estimator_.predict(X_test)

## after trusting my model
def evaluate_model(X_train, y_train, X_test, y_test, grid_search):
    grid_search.fit(X_train, y_train)

    y_test_pred = grid_search.predict(X_test)

    cm = ic(confusion_matrix(y_test, y_test_pred))
    ic(precision_score(y_test, y_test_pred))
    ic(recall_score(y_test, y_test_pred))
    ic(f1_score(y_test, y_test_pred))

    plt.matshow(cm, cmap=plt.cm.gray)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

    y_scores = grid_search.predict_proba(X_test)[:, 1]

    prec, rec, thresh = precision_recall_curve(y_test, y_scores)
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

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

evaluate_model(X_train, y_train, X_test, y_test, grid_search.best_estimator_)
```