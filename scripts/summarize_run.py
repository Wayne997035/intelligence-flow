"""Print a human-readable digest of data/latest_run.json for CI logs."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str = "data/latest_run.json") -> int:
    artifact = Path(path)
    if not artifact.exists():
        print(f"{path} not found")
        return 1
    data = json.loads(artifact.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    print("=== meta ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for key in ("stock_report", "ai_report"):
        report = data.get(key) or {}
        print(f"\n=== {key}: {report.get('title')} ===")
        print("summary:", (report.get("summary") or "")[:400])
        for index, item in enumerate(report.get("items", []), 1):
            print(f"{index:2d}. [{item.get('source_type')}] {item.get('published_at')} | {item.get('title')}")
        appendix = (report.get("metadata") or {}).get("appendix_items", [])
        print(f"-- appendix ({len(appendix)}) --")
        for item in appendix:
            print(f"  * [{item.get('source_type')}] {item.get('published_at')} | {item.get('title')}")
    quotes = (data.get("stock_payload") or {}).get("embeds", [{}])[0].get("description", "")
    print("\n=== stock quotes (discord) ===")
    print(quotes[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
