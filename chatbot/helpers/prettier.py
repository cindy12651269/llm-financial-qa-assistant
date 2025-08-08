# helpers/prettier.py
import json

def prettify_source(source: dict) -> str:
    score = source.get("score")
    src_label = source.get("source", "unknown")
    doc_label = source.get("document") or src_label
    org = source.get("organization", "")
    year = source.get("fiscal_year", "")
    rtype = source.get("report_type", "")
    sector = source.get("sector", "")
    preview = source.get("content_preview", "")

    pretty_preview = preview
    if isinstance(preview, str) and preview.strip().startswith("{"):
        try:
            pretty_preview = "```json\n" + json.dumps(json.loads(preview), indent=2)[:800] + "\n```"
        except Exception:
            pretty_preview = preview[:300] + "..."

    head = f"**Source:** {src_label} — score {score:.3f}" if isinstance(score, (float, int)) else f"**Source:** {src_label}"
    meta = []
    if org: meta.append(f"- Org: {org}")
    if year: meta.append(f"- FY: {year}")
    if rtype: meta.append(f"- Type: {rtype}")
    if sector: meta.append(f"- Sector: {sector}")
    meta_str = "  \n".join(meta)

    body = f"\n\n{pretty_preview}" if pretty_preview else ""
    return f"{head}\n\n{meta_str}\n{body}".rstrip()
