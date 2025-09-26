#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
data = load_wine()
X = data.data
y = data.target

def run_experiment(train_size, val_size=None, test_size=None, random_state=42):
    if val_size is not None:
        # test split first
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        # validation from remaining
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size/(1 - test_size),
            stratify=y_train_val, random_state=random_state
        )
    else:
        # train/test only (no validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        X_val, y_val = None, None

    # scale + logistic regression
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=random_state))
    model.fit(X_train, y_train)

    val_acc = None
    if X_val is not None:
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    return val_acc, test_acc

# --- Run the three scenarios ---
val_60_20, test_60_20 = run_experiment(train_size=0.60, val_size=0.20, test_size=0.20)
val_70_15, test_70_15 = run_experiment(train_size=0.70, val_size=0.15, test_size=0.15)
val_none,  test_80_20 = run_experiment(train_size=0.80, val_size=None,  test_size=0.20)

# Print results
print("=== Comparison of Splits ===")
print(f"60:20:20 -> Validation Accuracy: {val_60_20:.4f}, Test Accuracy: {test_60_20:.4f}")
print(f"70:15:15 -> Validation Accuracy: {val_70_15:.4f}, Test Accuracy: {test_70_15:.4f}")
print(f"80:20 (no val) -> Validation Accuracy: {val_none}, Test Accuracy: {test_80_20:.4f}")


# In[ ]:




