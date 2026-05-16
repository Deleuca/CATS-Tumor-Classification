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

_ANNOTATED_CALL = None


def _get_annotated_call(call_path=CALL_PATH):
    """Build (or return cached) per-probe gene annotation."""
    global _ANNOTATED_CALL
    if _ANNOTATED_CALL is None:
        call = pd.read_csv(call_path, sep="\t")
        bptogene = pd.read_csv("BasepairToGeneMap.tsv", delimiter="\t")
        call["Chromosome"] = call["Chromosome"].astype(str).replace({"23": "X"})
        call["Gene"] = call.apply(lambda row: _find_genes(row, bptogene), axis=1)
        _ANNOTATED_CALL = call
    return _ANNOTATED_CALL


def genes_for_features(feature_ids, call_path=CALL_PATH):
    """Return sorted list of HGNC symbols overlapping the given probe indices."""
    annotated = _get_annotated_call(call_path)
    genes = set()
    for fid in feature_ids:
        probe_genes = annotated.iloc[int(fid)]["Gene"]
        if probe_genes:
            genes |= set(probe_genes)
    return sorted(genes)


def select_bc_features(
    feature_ids,
    call_path=CALL_PATH,
    genes_path=GENES_PATH,
):
    if not os.path.isfile(GENES_PATH):
        get_breast_cancer_genes()
    with open(genes_path) as f:
        bc = {line.strip() for line in f if line.strip()}

    annotated = _get_annotated_call(call_path)
    kept, genes = [], set()
    for fid in tqdm(feature_ids, desc="Gene lookup"):
        probe_genes = annotated.iloc[int(fid)]["Gene"]
        if not probe_genes:
            continue
        hits = set(probe_genes) & bc
        if hits:
            kept.append(fid)
            genes |= hits
    return kept, sorted(genes)