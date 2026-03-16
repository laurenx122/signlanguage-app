import csv
import time
from typing import Optional
from pathlib import Path

class SentenceBuilder:
    def __init__(self,
                 labels_path = Path(__file__).parent.parent.parent / "data/raw/fsl_dynamic/labels.csv",
                 short_pause=0.8,
                 long_pause=2.2,
                 max_tokens=25):

        self.short_pause = short_pause
        self.long_pause = long_pause
        self.max_tokens = max_tokens

        self.tokens = []
        self.pause_start_time: Optional[float] = None

        self.categories = {}
        self.load_labels(labels_path)

        # Convenience sets
        self.qwords = self.categories.get("QUESTIONS", set())
        self.actions = self.categories.get("ACTIONS", set())
        self.people = self.categories.get("PEOPLE", set())
        self.answers = self.categories.get("ANSWERS", set())
        self.places = self.categories.get("PLACE", set())
        self.polite = self.categories.get("POLITENESS", set())
        self.time_words = self.categories.get("TIME", set())
        self.identity = self.categories.get("IDENTITY", set())

    # ------------------------------
    # LOAD LABELS
    # ------------------------------
    def load_labels(self, path):
        with open(path, encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                _, word, category = row
                word = word.upper()
                category = category.upper()
                if category not in self.categories:
                    self.categories[category] = set()
                self.categories[category].add(word)

    # ------------------------------
    # ADD TOKEN
    # ------------------------------
    def add_token(self, token: str):
        token = token.upper().strip()
        if not token:
            return None
        if self.tokens and self.tokens[-1] == token:
            return None
        self.tokens.append(token)

    # ------------------------------
    # UPDATE PAUSE (auto finalize)
    # ------------------------------
    def update_pause(self, hands_present: bool):
        """
        Call this every frame. If hands disappear for long_pause, finalize sentence.
        Returns: (raw_sentence, expanded_sentence) or None
        """
        now = time.time()
        if hands_present:
            self.pause_start_time = now
            return None

        if self.tokens:
            if self.pause_start_time is None:
                self.pause_start_time = now
            elapsed = now - self.pause_start_time
            if elapsed >= self.long_pause:
                raw, sentence = self.finalize()
                self.pause_start_time = None
                return raw, sentence
        return None

    # ------------------------------
    # FINALIZE
    # ------------------------------
    def finalize(self):
        raw = " ".join(self.tokens)
        sentence = self.expand(raw)
        self.tokens = []
        return raw, sentence

    # ------------------------------
    # GRAMMAR ENGINE
    # ------------------------------
    def expand(self, raw: str) -> str:
        toks = raw.split()
        if not toks:
            return ""

        greeting = None
        if toks[0] in self.polite:
            greeting = toks[0]
            toks = toks[1:]

        if not toks:
            return greeting.title() + "."

        # --------------------------------
        # QUESTION WORD SENTENCES
        # --------------------------------
        if toks[0] in self.qwords:
            if "NAME" in toks:
                sentence = "What is your name?"
            elif len(toks) >= 2 and toks[0] == "HOW" and toks[1] == "YOU":
                # Common connector for "How you" → "How are you?"
                sentence = "How are you?"
            else:
                sentence = " ".join(t.lower() for t in toks).capitalize() + "?"
            if greeting:
                sentence = greeting.title() + ", " + sentence.lower()
            return sentence

        # --------------------------------
        # YES / NO QUESTIONS
        # --------------------------------
        if toks[0] == "YOU":
            if len(toks) >= 2 and toks[1] in self.actions:
                verb = toks[1].lower()
                remainder = " ".join(t.lower() for t in toks[2:])
                if remainder:
                    return f"You {verb} {remainder}?"
                return f"You {verb}?"

        # --------------------------------
        # STATEMENTS
        # --------------------------------
        if toks[0] == "I":
            if len(toks) >= 2 and toks[1] in self.actions:
                verb = toks[1].lower()
                remainder = " ".join(t.lower() for t in toks[2:])
                if remainder:
                    return f"I {verb} {remainder}."
                return f"I {verb}."

        # --------------------------------
        # PEOPLE CONNECTION
        # --------------------------------
        if len(toks) == 2 and toks[0] in self.people and toks[1] in self.people:
            return f"{toks[0].title()} and {toks[1].lower()}."

        # --------------------------------
        # ANSWERS
        # --------------------------------
        if toks[0] in self.answers:
            return toks[0].title() + "."

        # --------------------------------
        # FALLBACK
        # --------------------------------
        return " ".join(t.title() for t in toks) + "."