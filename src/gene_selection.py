import requests
import pandas as pd
import os


CALL_PATH = "B4TM_CATS_training_data/Train_call.tsv"
GENES_PATH = "breast_cancer_genes.txt"
UCSC_URL = "https://api.genome.ucsc.edu/getData/track"


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


def _genes_in_region(chrom, start, end, genome="hg19", track="ncbiRefSeq"):
    """Return gene symbols overlapping chr<chrom>:<start>-<end> via UCSC REST."""
    params = {
        "genome": genome,
        "track": track,
        "chrom": f"chr{chrom}",
        "start": start,
        "end": end,
    }
    resp = requests.get(UCSC_URL, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get(track, [])
    return {it["name2"] for it in items if it.get("name2")}


def select_bc_features(
    feature_ids,
    call_path=CALL_PATH,
    genes_path=GENES_PATH,
    genome="hg19",
):
    if not os.path.isfile(GENES_PATH):
        get_breast_cancer_genes()
    call = pd.read_csv(call_path, sep="\t")
    with open(genes_path) as f:
        bc = {line.strip() for line in f if line.strip()}

    kept, genes = [], set()
    for fid in feature_ids:
        chrom, start, end = call.iloc[int(fid)][["Chromosome", "Start", "End"]]
        hits = _genes_in_region(int(chrom), int(start), int(end), genome=genome) & bc
        if hits:
            kept.append(fid)
            genes |= hits
    return kept, sorted(genes)
