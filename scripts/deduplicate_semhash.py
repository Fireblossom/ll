import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from semhash import SemHash  # type: ignore


def read_tsv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate TSV using SemHash.")
    parser.add_argument("--input", type=Path, required=True, help="Input TSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path for deduplicated rows")
    parser.add_argument(
        "--text-column",
        type=str,
        default="Features",
        help="Column name to use as text for deduplication",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        help="Optional list of columns to keep in output; defaults to all",
    )
    parser.add_argument(
        "--use-ann",
        action="store_true",
        help="Use ANN backend (recommended for larger datasets)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Semantic similarity threshold for considering duplicates (SemHash default applies if unset)",
    )
    args = parser.parse_args()

    rows = read_tsv(args.input)
    if not rows:
        print("No rows in input; exiting.")
        return

    if args.text_column not in rows[0]:
        raise ValueError(f"Text column '{args.text_column}' not found in TSV header: {list(rows[0].keys())}")

    # Select columns to keep
    fieldnames: List[str]
    if args.columns:
        fieldnames = args.columns
    else:
        fieldnames = list(rows[0].keys())

    # Build records for SemHash (list of dicts is supported)
    records: List[Dict[str, Any]] = rows

    # Initialize SemHash and run self-deduplication
    semhash = SemHash.from_records(records=records, columns=[args.text_column], use_ann=args.use_ann)
    if args.threshold is not None:
        result = semhash.self_deduplicate(threshold=args.threshold)
    else:
        result = semhash.self_deduplicate()

    selected: List[Dict[str, Any]] = result.selected if hasattr(result, "selected") else result

    # Keep only requested columns
    output_rows: List[Dict[str, Any]] = []
    for r in selected:
        out: Dict[str, Any] = {}
        for k in fieldnames:
            out[k] = r.get(k)
        output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output, output_rows, fieldnames)
    print(f"Input rows: {len(rows)} -> Deduplicated rows: {len(output_rows)}")
    print(f"Wrote deduplicated TSV to {args.output}")


if __name__ == "__main__":
    main()


