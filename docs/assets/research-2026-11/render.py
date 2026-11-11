"""Render conceptual README diagrams. Requires matplotlib; no project data used."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "osv-research-2026-11"
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
INK = "#162B36"
MUTED = "#506571"
TEAL = "#087F80"
BLUE = "#3468AA"
AMBER = "#9A641C"
BG = "#F6F8F8"


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text(.55, 8.45, title, fontsize=23, weight="bold", color=INK)
    ax.text(.55, 8.02, subtitle, fontsize=12, color=MUTED)
    ax.text(.55, .23, "OSV  /  NOVEMBER 2026     •     CONCEPTUAL DESIGN  /  NO EXPERIMENTAL RESULTS", fontsize=9, color=MUTED)
    return fig, ax


def box(ax, x, y, w, h, title, body, accent=TEAL, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.10", linewidth=1.2, edgecolor=accent, facecolor=fill))
    ax.text(x+.20, y+h-.38, title, fontsize=13, weight="bold", color=accent, va="top")
    ax.text(x+.20, y+h-.82, body, fontsize=11, color=INK, va="top", linespacing=1.6)


def arrow(ax, a, b, color=MUTED, style="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color=color, connectionstyle=style))


def save(fig, name):
    fig.savefig(OUT / (name+".png"), dpi=150, facecolor=BG)
    svg = OUT / (name+".svg")
    fig.savefig(svg, facecolor=BG, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def pipeline():
    fig, ax = canvas("Clean first. Preserve evidence. Compare cautiously.", "Selected early pipeline; learned components enter only after benchmark and permission checks.")
    box(ax, .55, 5.65, 2.65, 1.8, "01  QUERY", "Document → manual crop\nOptional stamp-region hint\nOriginal retained")
    box(ax, 3.8, 5.65, 3.0, 1.8, "02  REMOVE STAMP", "Conservative colour separation\nNo inpainting or reconstruction\nNo detection ≠ clean")
    box(ax, 7.4, 5.65, 3.25, 1.8, "03  REPRESENT", "Normalise image + masks\nClassical / DINOv2 candidate\nGlobal and patch features", BLUE)
    box(ax, 11.25, 5.65, 4.15, 1.8, "04  COMPARE", "Cleaned query against references\nGlobal baseline; local matching\nReliability from recorded provenance", BLUE)
    for a,b in [((3.2,6.55),(3.8,6.55)),((6.8,6.55),(7.4,6.55)),((10.65,6.55),(11.25,6.55))]: arrow(ax,a,b)
    box(ax, 3.8, 2.95, 3.0, 2.0, "PROVENANCE", "Original + hash; stamp estimate\nIntervention + actual changes\nInk support + ambiguous areas")
    arrow(ax,(5.3,5.65),(5.3,4.95),TEAL)
    ax.plot([6.8, 7.1, 7.1, 11.0], [4.3, 4.3, 5.25, 5.25], color=TEAL, linewidth=1.6)
    arrow(ax,(11.0,5.25),(11.25,5.95),TEAL)
    ax.text(7.4, 4.96, "Reliability / coverage", fontsize=10, color=TEAL)
    box(ax, 10.0, 2.95, 5.4, 1.65, "GENUINE REFERENCES", "Quality checks + same normalisation / encoder\nNever used to reconstruct the query", BLUE)
    arrow(ax,(13.3,4.6),(13.3,5.65),BLUE)
    box(ax, .55, .85, 4.55, 1.4, "SIMILARITY", "Shown only after the benchmark gate", BLUE)
    box(ax, 5.7, .85, 4.55, 1.4, "EVIDENCE COVERAGE", "How much usable evidence remains", TEAL)
    box(ax, 10.85, .85, 4.55, 1.4, "REVIEW REASONS", "Weak or ambiguous evidence stays visible", AMBER)
    ax.text(.55, 2.57, "SEPARATE OUTPUTS FOR LOCAL HUMAN REVIEW", fontsize=11, color=MUTED, weight="bold")
    ax.text(.55, 4.45, "Originals remain available\nfor human inspection.\nOnly the cleaned query\nenters product comparison.", fontsize=11, color=MUTED, linespacing=1.6)
    save(fig, "01-current-pipeline")


def hypotheses():
    fig, ax = canvas("Two testable research hypotheses", "Evaluate controlled alternatives on held-out writers and stamps; support, reject or report inconclusive.")
    box(ax,.55,4.05,7.15,3.45,"H1  RELIABILITY-AWARE LOCAL MATCHING", "Does local matching use stamp-affected evidence\nmore safely than global or equal-weight comparison?\n\nHold cleaner, encoder and data fixed.\nCompare at matched usable evidence.\nMeasure FAR at a fixed validation-selected FRR.", BLUE)
    box(ax,8.25,4.05,7.15,3.45,"H2  PROVENANCE-BASED REVIEW", "Do cleaning records help identify cases\nwhere the comparator is likely to fail?\n\nHold the comparator and cases fixed.\nCompare provenance-based vs score-only review.\nMeasure retained-case errors at the same review rate.", TEAL)
    box(ax,.55,1.05,4.55,2.3,"CLEANING", "Removal versus handwriting damage\nColour, overlap and scan conditions\nSynthetic + physical examples", AMBER)
    box(ax,5.7,1.05,4.55,2.3,"LEARNING + TRANSFER", "Frozen → learned representations\nUnseen writers and stamp designs\nPhysical-scan domain gap", BLUE)
    box(ax,10.85,1.05,4.55,2.3,"SHARED PROTOCOL", "Writer / source / stamp separation\nValidation before final testing\nPredicted vs oracle masks", TEAL)
    ax.text(.55,3.65,"SUPPORTING QUESTIONS AND CONTROLS",fontsize=11,color=MUTED,weight="bold")
    ax.text(.55,.67,"Uncleaned-query evaluation remains conditional on owner approval. Novelty is not established.",fontsize=10,color=MUTED)
    save(fig,"02-research-questions")


def roadmap():
    fig, ax = canvas("Deliver early. Learn from measured failures.", "One codebase; a useful customer prototype does not wait for the research programme.")
    box(ax,.55,3.7,4.55,3.75,"01  CUSTOMER PROTOTYPE", "M1a  Cleaner + complete provenance\nM1b  Cleaner comparison + physical tests\nM2a  Baselines + reference comparison\nM2b  Held-out benchmark / score gate\nM3    Local reviewer interface\n\nFirst useful delivery", TEAL)
    box(ax,5.7,3.7,4.55,3.75,"02  RESEARCH + LEARNING", "M4  Stamp ablations + review / H2\nM5  Reliable local matching / H1\nM6  Learned comparator\n        → own signature encoder\n        → stamp-aware training\n\nResults determine the next change", BLUE)
    box(ax,10.85,3.7,4.55,3.75,"03  CONDITIONAL EXTENSIONS", "M7  Learned stamp / ink segmentation\n        Move earlier if cleaning fails\nM8  Occlusion-aware verifier\n        Only if evidence supports it\n        Automatic detection if useful\n\nData, permissions and compute gated", AMBER)
    arrow(ax,(5.1,5.5),(5.7,5.5))
    arrow(ax,(10.25,5.5),(10.85,5.5))
    ax.text(.55,3.25,"PUBLIC EVIDENCE GROWS THROUGHOUT THE PROJECT",fontsize=11,color=MUTED,weight="bold")
    box(ax,.55,1.1,7.15,1.75,"M9  REPRODUCIBLE PORTFOLIO", "Demo + documented decisions + ablations + failure cases\nPermitted examples only; no inflated performance claims",TEAL)
    box(ax,8.25,1.1,7.15,1.75,"M10  JANUARY 2027 THESIS DISCUSSION", "Research question + related work + prototype + results\nScope and novelty to confirm with the professor",BLUE)
    ax.text(.55,.68,"Roadmap, not a fixed schedule. Milestones may span several sessions; optional work is not on the delivery path.",fontsize=10,color=MUTED)
    save(fig,"03-development-roadmap")


if __name__ == "__main__":
    pipeline()
    hypotheses()
    roadmap()
