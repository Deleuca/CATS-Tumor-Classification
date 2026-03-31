from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np

def read_data():
    call = pd.read_csv("B4TM_CATS_training_data/Train_call.tsv", sep="\t")
    clinical = pd.read_csv("B4TM_CATS_training_data/Train_clinical.tsv", sep="\t")

    clinical["Subgroup"] = clinical["Subgroup"].astype("category")

    sample_columns = call.columns[4:]
    transposed_call = call[sample_columns].T
    
    transposed_call.index.name = "Sample"
    transposed_call.reset_index(inplace=True)

    merged = pd.merge(transposed_call, clinical, on="Sample")

    transposed_call_clean = merged.drop(columns=["Sample", "Subgroup"])
    clinical_clean = merged["Subgroup"]

    return transposed_call_clean, clinical_clean

def random_forest(X,y):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    accuracy = cross_val_score(model, X, y, cv=5)
    return accuracy
