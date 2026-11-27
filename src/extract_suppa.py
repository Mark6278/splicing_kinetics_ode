import argparse, os, re
import pandas as pd


def get_target_ids(gtf_path: str, target: str) -> set[str]:
    key = target.strip().upper()
    ids = {key}
    if gtf_path and os.path.exists(gtf_path):
        with open(gtf_path) as f:
            for line in f:
                if not line.startswith("#") and "\tgene\t" in line:
                    m_name = re.search(r'gene_name "([^"]+)"', line, re.I)
                    m_id = re.search(r'gene_id "([^".]+)', line, re.I)
                    if m_name and m_id and (key == m_name.group(1).upper() or key == m_id.group(1).upper()):
                        ids.add(m_id.group(1).upper())
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", required=True)
    ap.add_argument("--gtf", default="")
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t_ids = get_target_ids(args.gtf, args.target)
    df = pd.read_csv(args.psi, sep="\t", index_col=0)
    df.index.name = "event_id"
    df = df.reset_index()
    df["gene"] = df["event_id"].apply(lambda x: re.split(r"[;.]", str(x)).pop(0).upper())
    matched = df[df["gene"].isin(t_ids)].copy()
    if matched.empty:
        raise SystemExit(f"No SUPPA2 events found for '{args.target}' in {args.psi}")
    matched["gene"] = args.target
    cols = ["event_id", "gene"] + [c for c in matched.columns if c not in ("event_id", "gene")]
    matched[cols].to_csv(args.output, index=False)
    print(f"Extracted {len(matched)} SUPPA2 events for '{args.target}' -> {args.output}")


if __name__ == "__main__":
    main()
