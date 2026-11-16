"""Reference/query sampling protocol for the clean benchmark (M2b).

For every writer of a partition (the *claimed writer*):

* **References** — the first ``k`` eligible images in the order
  ``sha256("<seed>:order:<writer>:<file sha256>")``.
* **Genuine queries** — all remaining eligible images of that writer.
* **Random impostor queries** — the first ``n`` other writers of the *same partition* in the
  order ``sha256("<seed>:impostor:<claimed>:<other>")``; from each, the one image of its genuine
  query pool that minimises ``sha256("<seed>:impostor-image:<claimed>:<other>:<file sha256>")``.
  So impostor queries against one claimed writer come from ``n`` different writers, and an
  impostor image is never one of its own writer's references.

Everything is derived from hashes of the seed and of file contents: no random state, no scores.
A "random impostor" is a genuine signature of somebody else presented under the claimed
identity. It is not a forgery attempt, so nothing here measures skilled-forgery rejection.

:func:`check_protocol` enforces reference/query separation (by path and by decoded pixels) and
that every comparison stays inside one partition.
"""

from __future__ import annotations

import hashlib

from osv.datasets.signature_folder import PARTITIONS, eligible_by_writer

PROTOCOL_SCHEMA = "osv-clean-protocol-1.0"
RULE = ("references: first k images by sha256('<seed>:order:<writer>:<file sha256>'); genuine queries: the rest; "
        "impostor queries: first n other writers of the same partition by sha256('<seed>:impostor:<claimed>:<other>'), "
        "one image each from that writer's genuine-query pool, minimising "
        "sha256('<seed>:impostor-image:<claimed>:<other>:<file sha256>')")


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_protocol(manifest: dict, splits: dict, k_references: int, n_impostors: int, seed: int) -> dict:
    if k_references < 1 or n_impostors < 0:
        raise ValueError("need at least one reference and a non-negative number of impostors")
    eligible = eligible_by_writer(manifest)
    partitions = {}
    for part in PARTITIONS:
        writers = sorted(splits["partitions"][part])
        pools: dict[str, dict] = {}
        for w in writers:
            images = sorted(eligible.get(w, []), key=lambda r: _h(f"{seed}:order:{w}:{r['sha256']}"))
            if len(images) <= k_references:
                raise ValueError(f"writer {w} has {len(images)} eligible images; needs more than k={k_references}")
            pools[w] = {"references": images[:k_references], "genuine": images[k_references:]}
        units = []
        for w in writers:
            others = sorted((o for o in writers if o != w), key=lambda o: _h(f"{seed}:impostor:{w}:{o}"))[:n_impostors]
            impostors = []
            for o in others:
                img = min(pools[o]["genuine"], key=lambda r: _h(f"{seed}:impostor-image:{w}:{o}:{r['sha256']}"))
                impostors.append({"writer": o, "path": img["path"], "sha256": img["sha256"]})
            units.append({
                "claimed_writer": w,
                "references": [{"path": r["path"], "sha256": r["sha256"]} for r in pools[w]["references"]],
                "genuine_queries": [{"path": r["path"], "sha256": r["sha256"]} for r in pools[w]["genuine"]],
                "impostor_queries": impostors,
            })
        partitions[part] = units
    return {
        "schema": PROTOCOL_SCHEMA,
        "rule": RULE,
        "seed": seed,
        "k_references": k_references,
        "n_impostors_per_writer": n_impostors,
        "impostor_type": "random impostor: a genuine signature of another writer of the same partition; not a forgery",
        "partitions": partitions,
        "counts": {p: {"writers": len(u), "genuine_comparisons": sum(len(x["genuine_queries"]) for x in u),
                       "impostor_comparisons": sum(len(x["impostor_queries"]) for x in u)}
                   for p, u in partitions.items()},
    }


def check_protocol(protocol: dict, splits: dict, manifest: dict) -> None:
    """Raise on any breach of reference/query separation or partition containment."""
    pixels = {r["path"]: r.get("pixel_sha256") for r in manifest["records"]}
    writer_of = {r["path"]: r["writer"] for r in manifest["records"]}
    excluded = {r["path"] for r in manifest["records"] if r["excluded"]}
    for part, units in protocol["partitions"].items():
        members = set(splits["partitions"][part])
        reference_paths = {r["path"] for u in units for r in u["references"]}
        for u in units:
            w = u["claimed_writer"]
            if w not in members:
                raise ValueError(f"{w} is not a writer of partition {part}")
            refs = {r["path"] for r in u["references"]}
            ref_pixels = {pixels[p] for p in refs}
            if len(refs) != len(u["references"]) or len(ref_pixels) != len(refs):
                raise ValueError(f"{w}: repeated reference image")
            for r in u["references"] + u["genuine_queries"]:
                if writer_of[r["path"]] != w:
                    raise ValueError(f"{w}: {r['path']} belongs to another writer")
            for q in u["genuine_queries"] + u["impostor_queries"]:
                if q["path"] in refs or pixels[q["path"]] in ref_pixels:
                    raise ValueError(f"{w}: query {q['path']} is also a reference")
            for q in u["impostor_queries"]:
                if q["writer"] == w or writer_of[q["path"]] != q["writer"] or q["writer"] not in members:
                    raise ValueError(f"{w}: impostor {q['path']} is not from another writer of {part}")
                if q["path"] in reference_paths:
                    raise ValueError(f"{w}: impostor {q['path']} is some writer's reference")
            for r in u["references"] + u["genuine_queries"] + u["impostor_queries"]:
                if r["path"] in excluded:
                    raise ValueError(f"{w}: {r['path']} is excluded by the manifest")
