import requests
import pandas as pd
import os
from tqdm import tqdm


CALL_PATH = "B4TM_CATS_training_data/Train_call.tsv"
GENES_PATH = "breast_cancer_genes.txt"


def get_breast_cancer_genes():
    url = "https://rest.kegg.jp/get/hsa05224"
    response = requests.get(url)

    lines = response.text.split("\n")
    genes = []
    record = False

    for line in lines:
        if line.startswith("GENE"):
            record = True
            genes.append(line[12:])
        elif record:
            if line.startswith(" "):
                genes.append(line[12:])
            else:
                break

    symbols = []

    for g in genes:
        symbol = g.split(";")[0].split()[1]
        symbols.append(symbol)

    with open(GENES_PATH, "w") as f:
        for s in symbols:
            f.write(s + "\n")


def _find_genes(row, genemap_df):
    """Given a call df row and a gene map, return a list of genes corresponding to that region"""
    overlaps = (
        (genemap_df["Chromosome"] == row["Chromosome"]) & 
        ((genemap_df["Gene_start"] <= row["End"]) & (genemap_df["Gene_end"] >= row["Start"]))
    )

    genes = genemap_df[overlaps]["HGNC_symbol"]
    if len(genes) > 0:
        return genes.astype(str).tolist()
    else:
        return ""

def select_bc_features(
    feature_ids,
    call_path=CALL_PATH,
    genes_path=GENES_PATH,
):
    if not os.path.isfile(GENES_PATH):
        get_breast_cancer_genes()
    call = pd.read_csv(call_path, sep="\t")
    with open(genes_path) as f:
        bc = {line.strip() for line in f if line.strip()}

    bptogene = pd.read_csv("BasepairToGeneMap.tsv", delimiter="\t")
    call["Chromosome"] = call["Chromosome"].astype(str).replace({"23": "X"})

    annotated_call = call.copy(deep=True)
    annotated_call["Gene"] = annotated_call.apply(lambda row: _find_genes(row, bptogene), axis=1)

    kept, genes = [], set()
    for fid in tqdm(feature_ids, desc="Gene lookup"):
        probe_genes = annotated_call.iloc[int(fid)]["Gene"]
        if not probe_genes:
            continue
        hits = set(probe_genes) & bc
        if hits:
            kept.append(fid)
            genes |= hits
    return kept, sorted(genes)