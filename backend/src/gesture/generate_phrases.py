# backend/src/gesture/generate_phrases.py
"""
Generate phrase_bank.csv from labels.csv + pair_rules.csv

- Reads labels from: backend/data/raw/fsl_dynamic/labels.csv
- Reads pair rules from: backend/src/gesture/CSV/pair_rules.csv
- Writes output to: backend/src/gesture/CSV/phrase_bank.csv

Output columns:
- phrase  -> raw tokens (what is signed)
- english -> best-effort grammatical English (fallback to raw if unsure)

Key features:
✅ Allows comments in pair_rules.csv using lines starting with '#'
✅ Generates:
   - single-token phrases (all labels)
   - pair phrases based on pair_rules.csv
   - small set of useful 3–4 token templates for conversation:
       * WHAT NAME
       * WHERE YOU LIVE
       * I WANT RED WHITE  (and similar)
       * PLEASE STOP
✅ Adds punctuation:
   - questions end with '?'
   - statements end with '.'
✅ Adds simple grammar:
   - color/food lists get "and"
   - I/YOU + DEAF -> am/are
   - TODAY/TOMORROW + weekday -> "Today is Monday."
   - I WANT X [X] -> "I want X (and Y)."
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


# ----------------------------
# Helpers
# ----------------------------
def norm(s: str) -> str:
    return s.strip().replace("’", "'").replace("‘", "'")


# ----------------------------
# Paths
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
    "QUESTION": ["QUESTION", "QUESTIONS"],
    "VERB": ["VERB", "VERBS"],
    "MODIFIER": ["MODIFIER", "MODIFIERS"],
    "IDENTITY": ["IDENTITY", "IDENTITIES"],
}


def cat(categories: Dict[str, List[str]], name: str) -> List[str]:
    key = name.upper()
    if key in categories:
        return categories[key]
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
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")

    categories: Dict[str, List[str]] = {}
    all_words: List[str] = []

    with LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

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

    # dedupe per category
    for k, items in categories.items():
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                out.append(x)
                seen.add(x)
        categories[k] = out

    # dedupe all_words
    seen = set()
    dedup_all = []
    for x in all_words:
        if x not in seen:
            dedup_all.append(x)
            seen.add(x)

    return categories, dedup_all


def load_pair_rules() -> List[Tuple[str, str]]:
    rules: List[Tuple[str, str]] = []
    if not PAIR_RULES_CSV.exists():
        return rules

    with PAIR_RULES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    for row in rows:
        if not row:
            continue
        first = (row[0] or "").strip()
        if not first:
            continue
        # allow comment lines
        if first.startswith("#"):
            continue
        # skip header
        if first.strip().upper() == "LEFT_CAT":
            continue
        if len(row) < 2:
            continue
        left = row[0].strip().upper()
        right = row[1].strip().upper()
        if left and right:
            rules.append((left, right))
    return rules


# ----------------------------
# English formatting (best-effort)
# ----------------------------
def to_english(tokens: List[str],
              colors: Set[str],
              foods: Set[str],
              days: Set[str],
              time_words: Set[str],
              greetings: Set[str],
              questions: Set[str],
              verbs: Set[str],
              modifiers: Set[str],
              identities: Set[str]) -> str:
    toks = [t for t in tokens if t]
    if not toks:
        return ""

    # collapse consecutive duplicates
    collapsed = [toks[0]]
    for t in toks[1:]:
        if t != collapsed[-1]:
            collapsed.append(t)
    toks = collapsed

    joined = " ".join(toks)

    # greetings (single)
    if joined in greetings:
        return joined.title()

    # TODAY/TOMORROW + weekday
    if len(toks) == 2 and toks[0] in time_words and toks[1] in days:
        return f"{toks[0].title()} is {toks[1].title()}."

    # THANK YOU / YOURE WELCOME exchange
    if joined == "THANK YOU YOURE WELCOME":
        return "Thank you. You're welcome."
    if joined == "YOURE WELCOME THANK YOU":
        return "You're welcome. Thank you."

    # I/YOU + DEAF
    if len(toks) == 2 and toks[0] in {"I", "YOU"} and toks[1] == "DEAF":
        subj = "I" if toks[0] == "I" else "You"
        verb = "am" if subj == "I" else "are"
        return f"{subj} {verb} deaf."

    # I/YOU + WRONG  (handle swapped order too)
    if len(toks) == 2 and "WRONG" in toks and ("I" in toks or "YOU" in toks):
        subj = "I" if "I" in toks else "You"
        verb = "am" if subj == "I" else "are"
        return f"{subj} {verb} wrong."

    # YES CORRECT / NO WRONG
    if joined == "YES CORRECT":
        return "Yes, correct."
    if joined == "NO WRONG":
        return "No, wrong."

    # Questions (simple)
    if toks[0] in questions:
        # WHAT NAME -> What is your name?
        if joined == "WHAT NAME":
            return "What is your name?"
        if joined == "WHO YOU":
            return "Who are you?"
        if joined == "WHERE YOU LIVE":
            return "Where do you live?"
        if len(toks) == 2 and toks[1] in identities:
            return f"{toks[0].title()} {toks[1].lower()}?"
        # fallback question
        return f"{' '.join([t.title() for t in toks])}?"

    # I/YOU + KNOW/UNDERSTAND
    if len(toks) == 2 and toks[0] in {"I", "YOU"} and toks[1] in {"KNOW", "UNDERSTAND"}:
        subj = "I" if toks[0] == "I" else "You"
        return f"{subj} {toks[1].lower()}."

    # PLEASE + VERB (Please stop.)
    if len(toks) == 2 and toks[0] == "PLEASE" and toks[1] in verbs:
        return f"Please {toks[1].lower()}."

    # I/YOU + WANT/LIKE/LOVE + X (+Y)
    if len(toks) in (3, 4) and toks[0] in {"I", "YOU"} and toks[1] in {"WANT", "LIKE", "LOVE"}:
        subj = "I" if toks[0] == "I" else "You"
        v = toks[1].lower()

        objs = toks[2:]
        if len(objs) == 1:
            return f"{subj} {v} {objs[0].lower()}."
        if len(objs) == 2:
            # add "and" between two objects (colors/foods/etc.)
            return f"{subj} {v} {objs[0].lower()} and {objs[1].lower()}."
        # fallback
        return f"{subj} {v} " + " ".join([o.lower() for o in objs]) + "."

    # FOOD pairs -> "Rice and egg."
    if len(toks) == 2 and toks[0] in foods and toks[1] in foods:
        if toks[0] == toks[1]:
            return f"{toks[0].title()}."
        return f"{toks[0].title()} and {toks[1].title()}."

    # COLOR pairs -> "Red and white."
    if len(toks) == 2 and toks[0] in colors and toks[1] in colors:
        if toks[0] == toks[1]:
            return f"{toks[0].title()}."
        return f"{toks[0].title()} and {toks[1].title()}."

    # HOT/COLD COFFEE
    if len(toks) == 2 and toks[1] == "COFFEE" and toks[0] in {"HOT", "COLD"}:
        return f"{toks[0].title()} coffee."

    # Default fallback: just show what was signed (title-cased) + period
    return f"{' '.join([t.title() for t in toks])}."


# ----------------------------
# Phrase generation
# ----------------------------
def add_phrase(phrases: Set[Tuple[str, str]], raw_tokens: List[str], english: str) -> None:
    raw = norm(" ".join([t for t in raw_tokens if t and t.strip()]))
    if not raw:
        return
    phrases.add((raw, english))


def main() -> None:
    categories, all_words = load_labels()
    pair_rules = load_pair_rules()

    # sets for english formatting
    colors = set(cat(categories, "COLOR"))
    foods = set(cat(categories, "FOOD"))
    days = set([w for w in all_words if w in ("MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY")])
    time_words = set([w for w in all_words if w in ("TODAY", "TOMORROW")])
    greetings = set(cat(categories, "GREETING"))
    questions = set(cat(categories, "QUESTION"))
    verbs = set(cat(categories, "VERB"))
    modifiers = set(cat(categories, "MODIFIER"))
    identities = set(cat(categories, "IDENTITY"))

    rel = cat(categories, "RELATIONSHIPS")

    phrases: Set[Tuple[str, str]] = set()

    # A) singles
    for w in all_words:
        eng = to_english([w], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
        # ensure single outputs end nicely
        if eng and not eng.endswith((".", "!", "?", ",")) and w not in greetings:
            eng = eng + "."
        add_phrase(phrases, [w], eng)

    # B) 2-token phrases from pair rules
    for left_cat, right_cat in pair_rules:
        left = cat(categories, left_cat)
        right = cat(categories, right_cat)
        if not left or not right:
            continue
        for a in left:
            for b in right:
                eng = to_english([a, b], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
                add_phrase(phrases, [a, b], eng)

    # C) small set of conversation templates (3–4 tokens)
    # 1) WHERE YOU LIVE, WHO YOU, WHAT NAME
    if "WHERE" in questions and "YOU" in rel and "LIVE" in verbs:
        eng = to_english(["WHERE", "YOU", "LIVE"], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
        add_phrase(phrases, ["WHERE", "YOU", "LIVE"], eng)

    if "WHO" in questions and "YOU" in rel:
        eng = to_english(["WHO", "YOU"], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
        add_phrase(phrases, ["WHO", "YOU"], eng)

    if "WHAT" in questions and "NAME" in identities:
        eng = to_english(["WHAT", "NAME"], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
        add_phrase(phrases, ["WHAT", "NAME"], eng)

    # 2) PLEASE STOP
    if "PLEASE" in greetings and "STOP" in verbs:
        eng = to_english(["PLEASE", "STOP"], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
        add_phrase(phrases, ["PLEASE", "STOP"], eng)

    # 3) I WANT X and I WANT X Y (colors/foods/drinks)
    if "I" in rel and "WANT" in verbs:
        # single object
        for obj in list(colors | foods | set(cat(categories, "DRINK"))):
            eng = to_english(["I", "WANT", obj], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
            add_phrase(phrases, ["I", "WANT", obj], eng)

        # two objects (especially colors + colors)
        for c1 in colors:
            for c2 in colors:
                if c1 == c2:
                    continue
                eng = to_english(["I", "WANT", c1, c2], colors, foods, days, time_words, greetings, questions, verbs, modifiers, identities)
                add_phrase(phrases, ["I", "WANT", c1, c2], eng)

    # Save
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase", "english"])
        for raw, eng in sorted(phrases, key=lambda x: x[0]):
            w.writerow([raw, eng])

    print("✅ LABELS_CSV:", LABELS_CSV.resolve(), "| exists:", LABELS_CSV.exists())
    print("✅ PAIR_RULES_CSV:", PAIR_RULES_CSV.resolve(), "| exists:", PAIR_RULES_CSV.exists())
    print("✅ OUTPUT_CSV:", OUTPUT_CSV.resolve())
    print("✅ Categories found:", ", ".join(sorted(categories.keys())))
    print("✅ Loaded pair rules:", len(pair_rules))
    print("✅ Generated phrases:", len(phrases))


if __name__ == "__main__":
    main()