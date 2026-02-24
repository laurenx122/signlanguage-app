# backend/src/gesture/generate_phrases.py
"""
Generate phrase_bank.csv from labels.csv + optional pair_rules.csv.

- Reads labels from: backend/data/raw/fsl_dynamic/labels.csv
- Reads pair rules from: backend/src/gesture/CSV/pair_rules.csv
- Writes phrase bank to: backend/src/gesture/CSV/phrase_bank.csv

Features:
✅ Always outputs single-word phrases (all labels)
✅ Generates useful default pairs even if pair_rules.csv is missing/empty:
   - TODAY/TOMORROW + weekdays
   - GREETING + GREETING (e.g., GOOD MORNING THANK YOU)
   - FOOD + FOOD (e.g., RICE EGG)
   - COLOR + COLOR (e.g., VIOLET PINK)
   - RELATIONSHIPS + RELATIONSHIPS (e.g., I YOU)
   - RELATIONSHIPS + SURVIVAL (e.g., I KNOW)
   - YES CORRECT, NO WRONG (if present)
✅ Uses category aliases so pair_rules can use canonical names even if labels.csv categories vary
✅ Auto-creates CSV folder
✅ (Optional) Auto-creates a starter pair_rules.csv if missing
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set


# ----------------------------
# Helpers
# ----------------------------
def norm(s: str) -> str:
    """Normalize text for consistency (smart apostrophes etc.)."""
    return s.strip().replace("’", "'").replace("‘", "'")


# ----------------------------
# Paths (based on your project structure)
# ----------------------------
GESTURE_DIR = Path(__file__).resolve().parent          # backend/src/gesture
BACKEND_ROOT = GESTURE_DIR.parents[1]                  # backend

LABELS_CSV = BACKEND_ROOT / "data" / "raw" / "fsl_dynamic" / "labels.csv"

CSV_DIR = GESTURE_DIR / "CSV"
CSV_DIR.mkdir(exist_ok=True)

PAIR_RULES_CSV = CSV_DIR / "pair_rules.csv"
OUTPUT_CSV = CSV_DIR / "phrase_bank.csv"


# ----------------------------
# Category aliasing
# ----------------------------
CATEGORY_ALIASES = {
    "GREETING": ["GREETING", "GREETINGS"],
    "SURVIVAL": ["SURVIVAL"],
    "DAYS": ["DAYS", "DAY"],
    "FAMILY": ["FAMILY", "FAMILIES"],
    "RELATIONSHIPS": ["RELATIONSHIPS", "RELATIONSHIP", "RELATION"],
    "COLOR": ["COLOR", "COLORS", "COLOUR", "COLOURS"],
    "FOOD": ["FOOD", "FOODS"],
    "DRINK": ["DRINK", "DRINKS", "BEVERAGE", "BEVERAGES"],
}


def cat(categories: Dict[str, List[str]], name: str) -> List[str]:
    """
    Fetch category list using canonical name or aliases.
    Example: cat(categories, "COLOR") will work even if labels.csv uses "COLORS".
    """
    key = name.upper()
    if key in categories:
        return categories[key]

    # Try alias group match
    for canonical, aliases in CATEGORY_ALIASES.items():
        if key == canonical or key in [a.upper() for a in aliases]:
            for a in aliases:
                a = a.upper()
                if a in categories:
                    return categories[a]
    return []


# ----------------------------
# Loading
# ----------------------------
def load_labels() -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Expected labels.csv rows: id,label,category
    Example:
      0,GOOD MORNING,GREETING
      10,UNDERSTAND,SURVIVAL
    """
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")

    categories: Dict[str, List[str]] = {}
    all_words: List[str] = []

    with LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Detect header
    start = 1 if rows and rows[0] and rows[0][0].lower() in ("id", "index") else 0

    for row in rows[start:]:
        if len(row) < 3:
            continue

        _, label, cat_name = row[0], row[1], row[2]
        label = norm(label)
        cat_name = norm(cat_name).upper()

        if not label or not cat_name:
            continue

        categories.setdefault(cat_name, []).append(label)
        all_words.append(label)

    # Dedupe per category, keep order
    for k, items in categories.items():
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                out.append(x)
                seen.add(x)
        categories[k] = out

    # Dedupe all words, keep order
    seen = set()
    dedup_all = []
    for x in all_words:
        if x not in seen:
            dedup_all.append(x)
            seen.add(x)

    return categories, dedup_all


def load_pair_rules() -> List[Tuple[str, str]]:
    """
    pair_rules.csv:
      LEFT_CAT,RIGHT_CAT
      GREETING,GREETING
      RELATIONSHIPS,SURVIVAL
    """
    rules: List[Tuple[str, str]] = []
    if not PAIR_RULES_CSV.exists():
        return rules

    with PAIR_RULES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    start = 1 if rows and rows[0] and rows[0][0].strip().upper() == "LEFT_CAT" else 0

    for row in rows[start:]:
        if len(row) < 2:
            continue
        left = row[0].strip().upper()
        right = row[1].strip().upper()
        if left and right:
            rules.append((left, right))
    return rules


def ensure_pair_rules_exists(categories: Dict[str, List[str]]) -> None:
    """
    Optional convenience: create a starter pair_rules.csv if missing.
    This prevents "phrase bank is too small" when you forgot to make rules.
    """
    if PAIR_RULES_CSV.exists():
        return

    def has(cat_name: str) -> bool:
        return len(cat(categories, cat_name)) > 0

    default_rules: List[Tuple[str, str]] = []

    # Reasonable defaults based on what's present
    if has("GREETING"):
        default_rules += [("GREETING", "GREETING"), ("GREETING", "FAMILY"), ("GREETING", "RELATIONSHIPS")]

    if has("RELATIONSHIPS"):
        default_rules += [
            ("RELATIONSHIPS", "RELATIONSHIPS"),
            ("RELATIONSHIPS", "SURVIVAL"),
            ("RELATIONSHIPS", "FAMILY"),
            ("RELATIONSHIPS", "FOOD"),
            ("RELATIONSHIPS", "DRINK"),
            ("RELATIONSHIPS", "COLOR"),
            ("RELATIONSHIPS", "DAYS"),
        ]

    if has("SURVIVAL"):
        default_rules += [("SURVIVAL", "SURVIVAL")]

    if has("FOOD"):
        default_rules += [("FOOD", "FOOD")]

    if has("COLOR"):
        default_rules += [("COLOR", "COLOR")]

    if has("DAYS"):
        default_rules += [("DAYS", "DAYS")]

    with PAIR_RULES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEFT_CAT", "RIGHT_CAT"])
        for a, b in default_rules:
            w.writerow([a, b])

    print(f"📝 Created starter pair_rules.csv at: {PAIR_RULES_CSV}")


# ----------------------------
# Generation
# ----------------------------
def add_phrase(phrases: Set[str], tokens: List[str]) -> None:
    phrase = " ".join([t for t in tokens if t and t.strip()]).strip()
    phrase = norm(phrase)
    if phrase:
        phrases.add(phrase)


def generate_default_pairs(categories: Dict[str, List[str]], all_words: List[str]) -> Set[str]:
    """
    Default useful pairs even without pair_rules.csv.
    These match your requested needs.
    """
    phrases: Set[str] = set()

    GREETING = cat(categories, "GREETING")
    FOOD = cat(categories, "FOOD")
    COLOR = cat(categories, "COLOR")
    REL = cat(categories, "RELATIONSHIPS")
    SURV = cat(categories, "SURVIVAL")

    # Detect TIME words from labels (since TODAY/TOMORROW are in DAYS in your data)
    TIME = [w for w in all_words if w in ("TODAY", "TOMORROW")]
    WEEKDAYS = [w for w in all_words if w in ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY")]

    # 1) TODAY/TOMORROW + weekday
    for t in TIME:
        for d in WEEKDAYS:
            add_phrase(phrases, [t, d])

    # 2) GREETING + GREETING (GOOD MORNING THANK YOU, etc.)
    for g1 in GREETING:
        for g2 in GREETING:
            if g1 != g2:
                add_phrase(phrases, [g1, g2])

    # Force THANK YOU ↔ YOURE WELCOME if present
    if "THANK YOU" in all_words and "YOURE WELCOME" in all_words:
        add_phrase(phrases, ["THANK YOU", "YOURE WELCOME"])
        add_phrase(phrases, ["YOURE WELCOME", "THANK YOU"])

    # 3) FOOD pairs
    for f1 in FOOD:
        for f2 in FOOD:
            if f1 != f2:
                add_phrase(phrases, [f1, f2])

    # 4) COLOR pairs
    for c1 in COLOR:
        for c2 in COLOR:
            if c1 != c2:
                add_phrase(phrases, [c1, c2])

    # 5) REL pairs (I YOU, YOU I)
    for r1 in REL:
        for r2 in REL:
            if r1 != r2:
                add_phrase(phrases, [r1, r2])

    # 6) YES CORRECT, NO WRONG
    if "YES" in all_words and "CORRECT" in all_words:
        add_phrase(phrases, ["YES", "CORRECT"])
    if "NO" in all_words and "WRONG" in all_words:
        add_phrase(phrases, ["NO", "WRONG"])

    # 7) REL + SURV (I KNOW, YOU UNDERSTAND, etc.)
    for r in REL:
        for s in SURV:
            add_phrase(phrases, [r, s])

    return phrases


def main() -> None:
    categories, all_words = load_labels()

    # Optional: auto-create starter pair_rules.csv if missing
    ensure_pair_rules_exists(categories)

    pair_rules = load_pair_rules()

    phrases: Set[str] = set()

    # A) Single words
    for w in all_words:
        add_phrase(phrases, [w])

    # B) Default useful pairs
    phrases |= generate_default_pairs(categories, all_words)

    # C) pair_rules-based pairs
    for left_cat, right_cat in pair_rules:
        left = cat(categories, left_cat)
        right = cat(categories, right_cat)
        if not left or not right:
            continue
        for a in left:
            for b in right:
                add_phrase(phrases, [a, b])

    # Save
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase"])
        for p in sorted(phrases):
            w.writerow([p])

    # Debug summary
    print("✅ LABELS_CSV:", LABELS_CSV.resolve(), "| exists:", LABELS_CSV.exists())
    print("✅ PAIR_RULES_CSV:", PAIR_RULES_CSV.resolve(), "| exists:", PAIR_RULES_CSV.exists())
    print("✅ OUTPUT_CSV:", OUTPUT_CSV.resolve())
    print("✅ Categories found:", ", ".join(sorted(categories.keys())))
    print("✅ Loaded pair rules:", len(pair_rules), "| sample:", pair_rules[:5])
    print("✅ Generated phrases:", len(phrases))


if __name__ == "__main__":
    main()