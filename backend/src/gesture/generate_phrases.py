# generate_phrases.py
import csv

GREETINGS = ["GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "THANK YOU", "YOURE WELCOME"]
SURVIVAL = ["UNDERSTAND", "KNOW", "DON’T KNOW", "NO", "YES", "WRONG", "CORRECT"]
DAYS = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
TIME = ["TODAY","TOMORROW"]
FAMILY = ["PARENTS"]
REL = ["I","YOU","DEAF"]
COLORS = ["BLUE","RED","WHITE","YELLOW","ORANGE","PINK","VIOLET"]
FOOD = ["BREAD","EGG","RICE","LONGANISA"]
DRINK = ["COFFEE"]
TEMP = ["HOT","COLD"]

templates = []

# Greetings
templates += [(g,) for g in GREETINGS]
templates += [(g1, g2) for g1 in GREETINGS for g2 in ["THANK YOU", "YOURE WELCOME"] if g1 != g2]

# Know/understand
templates += [("I", v) for v in ["KNOW","DON’T KNOW","UNDERSTAND"]]
templates += [("YOU", v) for v in ["KNOW","DON’T KNOW","UNDERSTAND"]]
templates += [("I","KNOW","YOU")]
templates += [("I","KNOW","YOU","DEAF")]
templates += [("YOU","DEAF")]

# Correctness
templates += [("YES","CORRECT"), ("NO","WRONG"), ("CORRECT",), ("WRONG",)]

# Days
templates += [(t, d) for t in TIME for d in DAYS]  # TODAY MONDAY, TOMORROW FRIDAY, etc.

# Colors
templates += [(c,) for c in COLORS]

# Food combos (simple)
templates += [(f,) for f in FOOD]
templates += [(f1, f2) for f1 in FOOD for f2 in FOOD if f1 != f2]

# Drinks
templates += [(temp, "COFFEE") for temp in TEMP]
templates += [("COFFEE",)]

# Family
templates += [("PARENTS",), ("I","PARENTS"), ("YOU","PARENTS")]

phrases = [" ".join(t) for t in templates]

with open("phrase_bank.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["phrase"])
    for p in sorted(set(phrases)):
        w.writerow([p])

print("Saved:", len(set(phrases)), "phrases to phrase_bank.csv")