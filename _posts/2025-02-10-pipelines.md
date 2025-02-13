---
title: 'How Did I Learned Pipelines for DS Projects'
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
we have a few upfronts we need to take care of:
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

we identify the numeric and categorical columns
```python
train_set_cat = ['ocean_proximity']
train_set_num = list(train_set.drop(columns=['ocean_proximity']).columns)
```

use preprocessor ColumnTransformet
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
  ('num', num_pipeline, train_set_num),
  ('cat', cat_pipeline, train_set_cat)
])
```

and finally fit_transform
```python
transform_train_set = preprocessor.fit_transform(train_set)
```
![My helpful screenshot](/images/WhatsApp_Image_2025-02-10.jpg)


and a brief from the book Hands On Machine Learning:
![My helpful screenshot](/images/1739196533941.jpg)


we can also define our own **custom Transformer** if for example we want to add a new column 
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


to the moment we all waited for, the score 🥳
```python
pipefinal.score(X_test, y_test)
```

how to save the pipeline
```python
import joblib
joblib.dump(pipefinal, 'pipe.joblib')

#to load
pipeline = joblib.load('pipe.joblib')
```

## Source: Medium Article
[Scikit-learn Pipelines Explained: Streamline and Optimize Your Machine Learning Processes \| by Sahin Ahmed, Data Scientist \| Medium](https://medium.com/@sahin.samia/scikit-learn-pipelines-explained-streamline-and-optimize-your-machine-learning-processes-f17b1beb86a4)

> [!Tip]
> other option for `ColumnTransformer` is `FeatureUnion` which works similarly but is more flexible, allowing any number of pipelines to be combined, not just column-wise operations.

![My helpful screenshot](/images/1739200574343.jpg)


