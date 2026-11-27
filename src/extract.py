import argparse
import json
from pathlib import Path
import pandas as pd


def resolve_target_queries(target: str) -> tuple[list[str], bool]:
    """
    O(1) lookup using src/human_gene_ensembl_mapping.json.
    Returns (search_terms, mapped_to_ensembl).
    """
    clean = target.strip().upper()
    queries = [clean]

    if clean.startswith("ENSG"):
        return queries, True

    map_file = Path(__file__).resolve().parent / "human_gene_ensembl_mapping.json"
    if map_file.exists():
        with open(map_file, "r", encoding="utf-8") as f:
            gene_map = json.load(f)

        ens_id = gene_map.get(clean)
        if ens_id:
            ens_up = str(ens_id).strip().upper()
            if ens_up not in queries:
                queries.append(ens_up)
            return queries, True

    return queries, False


def detect_separator(filepath: Path) -> str:
    name = filepath.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def main():
    parser = argparse.ArgumentParser(
        description="Universal chunked extractor for public PSI splicing tables using offline gene mapping."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to PSI table (.tsv, .csv, or .gz)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SRSF3",
        help="Gene symbol (e.g., SRSF3) or Ensembl ID (ENSG...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to target_data/<TARGET>_splicing_psi.csv)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    clean_target = args.target.strip().upper()
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"target_data/{clean_target}_splicing_psi.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = detect_separator(in_path)

    search_terms, mapped = resolve_target_queries(clean_target)
    if not mapped:
        print(
            f"Note: '{clean_target}' was not found in human_gene_ensembl_mapping.json. "
            f"Searching by symbol only—if no matches appear, re-run with its Ensembl Gene ID (e.g., --target ENSG00000112081)."
        )

    print(f"Searching {in_path.name} for: {', '.join(search_terms)}")

    matched_chunks = []
    for chunk in pd.read_csv(in_path, sep=sep, comment="#", chunksize=10000, low_memory=False):
        col_indices = [
            i for i, col_name in enumerate(chunk.columns)
            if any(k in str(col_name).lower() for k in ("gene", "symbol", "lsv", "event"))
        ]
        if not col_indices:
            col_indices = list(range(min(2, len(chunk.columns))))

        mask = pd.Series(False, index=chunk.index)
        for idx in col_indices:
            series_upper = chunk.iloc[:, idx].astype(str).str.upper()
            for term in search_terms:
                mask |= series_upper.str.contains(term, na=False, regex=False)

        filtered = chunk[mask]
        if not filtered.empty:
            matched_chunks.append(filtered)

    if not matched_chunks:
        print(
            f"No matches found for '{clean_target}' in {in_path.name}.\n"
            f"Tip: If you used a gene symbol or alias, look up its Ensembl Gene ID (ENSG...) and re-run with --target ENSG..."
        )
        return

    df_sub = pd.concat(matched_chunks, ignore_index=True)
    df_sub.to_csv(out_path, index=False)
    print(f"Extracted {len(df_sub)} rows for '{clean_target}' -> {out_path}")


if __name__ == "__main__":
    main()
    