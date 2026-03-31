from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np

#I'm not sure if the prints are necessary it was just to get an idea

call = pd.read_csv("B4TM_CATS_training_data/Train_call.tsv", sep="\t")
clinical = pd.read_csv("B4TM_CATS_training_data/Train_clinical.tsv", sep="\t")

print("Call matrix shape:", call.shape)
print(call.head())


print("Clinical label distribution:")
print(clinical["Subgroup"].value_counts())

#check if there are any missing values: (both are 0 so we're good to go)
#(total in the maxtrix)
print("missing values?", call.isnull().sum().sum())
#(per column)
print(clinical.isnull().sum())

#change the type to categorical
clinical["Subgroup"] = clinical["Subgroup"].astype("category")



# we take away the 4th first rows and transpose the call matrix :
sample_columns = call.columns[4:]
transposed_call = call[sample_columns].T
print(transposed_call.shape)
print(transposed_call.index[:5])
  
#rename
transposed_call.index.name = "Sample"
transposed_call.reset_index(inplace=True)

#because we're not sure the order is right, we merge the data files and check that
merged = pd.merge(transposed_call, clinical, on="Sample")

transposed_call_clean = merged.drop(columns=["Sample", "Subgroup"])
clinical_clean = merged["Subgroup"]

#now we can do the random forest with the cross validation
model = RandomForestClassifier(n_estimators=200, random_state=42)

def random_forest(X,y):
    #cross validation 
    accuracy = cross_val_score(model, X, y, cv=5)
    return accuracy

random_forest(transposed_call_clean, clinical_clean)
