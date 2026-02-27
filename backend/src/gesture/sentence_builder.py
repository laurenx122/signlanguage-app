# backend/src/gesture/sentence_builder.py
import time
from typing import List, Optional, Tuple


class SentenceBuilder:
    """
    Collect recognized tokens and finalize based on pauses.
    Then expand gloss-like tokens into more natural English.
    """

    def __init__(self, short_pause=0.8, long_pause=2.2, max_tokens=25):
        self.short_pause = short_pause
        self.long_pause = long_pause
        self.max_tokens = max_tokens

        self.tokens: List[str] = []
        self.pause_start_time: Optional[float] = None
        self.last_token_time: Optional[float] = None

        # categories (from your labels.csv)
        self.colors = {"RED", "WHITE", "YELLOW", "ORANGE", "PINK", "VIOLET"}
        self.foods = {"BREAD", "EGG", "RICE", "LONGANISA"}
        self.days = {"MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"}
        self.time_words = {"TODAY", "TOMORROW"}
        self.subjects = {"I", "YOU"}
        self.verbs = {"WANT", "EAT", "DRINK", "LIVE", "LIKE", "LOVE", "STOP", "KNOW", "UNDERSTAND"}
        self.qwords = {"WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW"}

        self.greetings = {
            "GOOD MORNING","GOOD AFTERNOON","GOOD EVENING",
            "THANK YOU","YOURE WELCOME","SORRY","PLEASE","BYE","OK"
        }

        self.ignore_tokens = {"WAITING", "COLLECTING...", "BUFFERING...", "UNKNOWN", ""}

    # ---------------------------
    # Token collection
    # ---------------------------
    def add_token(self, token: str) -> Optional[Tuple[str, str]]:
        now = time.time()
        token = token.strip().upper()

        if token in self.ignore_tokens:
            return None

        # prevent immediate duplicates
        if self.tokens and self.tokens[-1] == token:
            self.last_token_time = now
            return None

        self.tokens.append(token)
        self.last_token_time = now

        if len(self.tokens) >= self.max_tokens:
            return self.finalize()

        return None

    def update_pause(self, hands_detected: bool) -> Optional[Tuple[str, str]]:
        now = time.time()

        if hands_detected:
            self.pause_start_time = None
            return None

        if self.pause_start_time is None:
            self.pause_start_time = now
            return None

        pause_dur = now - self.pause_start_time
        if pause_dur >= self.long_pause and self.tokens:
            return self.finalize()

        return None

    def finalize(self) -> Tuple[str, str]:
        raw = " ".join(self.tokens).strip()
        eng = self.expand(raw)
        self.tokens = []
        self.pause_start_time = None
        self.last_token_time = None
        return raw, eng

    # ---------------------------
    # NEW: Canonicalization
    # ---------------------------
    def _canonicalize(self, toks: List[str]) -> List[str]:

        # remove consecutive duplicates
        toks = self._dedupe_consecutive(toks)

        # 1) PLEASE first
        if "PLEASE" in toks and toks[0] != "PLEASE":
            toks.remove("PLEASE")
            toks.insert(0, "PLEASE")

        # 2) Question word first
        for q in self.qwords:
            if q in toks and toks[0] != q:
                toks.remove(q)
                toks.insert(0, q)
                break

        # WHAT NAME (YOU) -> WHAT NAME
        if "WHAT" in toks and "NAME" in toks:
            return ["WHAT", "NAME"]

        # WHERE YOU LIVE (any order)
        if "WHERE" in toks and "YOU" in toks and "LIVE" in toks:
            return ["WHERE", "YOU", "LIVE"]

        # WHO YOU
        if "WHO" in toks and "YOU" in toks:
            return ["WHO", "YOU"]

        # TODAY/TOMORROW + weekday
        for t in self.time_words:
            if t in toks:
                for d in self.days:
                    if d in toks:
                        return [t, d]

        # Subject–Verb order (SVO)
        subj = next((t for t in toks if t in self.subjects), None)
        verb = next((t for t in toks if t in self.verbs), None)

        if subj and verb:
            # remove and reinsert properly
            toks = [t for t in toks if t not in {subj, verb}]
            toks.insert(0, subj)
            toks.insert(1, verb)

        return toks

    # ---------------------------
    # Expansion (gloss -> English)
    # ---------------------------
    def expand(self, raw: str) -> str:
        toks = [t for t in raw.split() if t]
        if not toks:
            return ""

        # Apply canonical grammar correction first
        toks = self._canonicalize(toks)

        joined = " ".join(toks)

        # Greetings
        if joined in self.greetings:
            return self._to_sentence(joined.title(), punct="")

        # TODAY MONDAY
        if len(toks) == 2 and toks[0] in self.time_words and toks[1] in self.days:
            return f"{toks[0].title()} is {toks[1].title()}."

        # THANK YOU <-> YOURE WELCOME
        if joined == "THANK YOU YOURE WELCOME":
            return "Thank you. You're welcome."
        if joined == "YOURE WELCOME THANK YOU":
            return "You're welcome. Thank you."

        # I/YOU DEAF
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] == "DEAF":
            verb = "am" if toks[0] == "I" else "are"
            return f"{toks[0].title()} {verb} deaf."

        # I/YOU WRONG
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] == "WRONG":
            verb = "am" if toks[0] == "I" else "are"
            return f"{toks[0].title()} {verb} wrong."

        # YES CORRECT / NO WRONG
        if joined == "YES CORRECT":
            return "Yes, correct."
        if joined == "NO WRONG":
            return "No, wrong."

        # I/YOU KNOW / UNDERSTAND
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] in {"KNOW","UNDERSTAND"}:
            return f"{toks[0].title()} {toks[1].lower()}."

        # I/YOU WANT X
        if len(toks) >= 3 and toks[0] in self.subjects and toks[1] in {"WANT","LIKE","LOVE"}:
            objs = toks[2:]
            if len(objs) == 1:
                return f"{toks[0].title()} {toks[1].lower()} {objs[0].lower()}."
            if len(objs) == 2:
                return f"{toks[0].title()} {toks[1].lower()} {objs[0].lower()} and {objs[1].lower()}."
            return f"{toks[0].title()} {toks[1].lower()} " + " ".join(o.lower() for o in objs) + "."

        # Color pairs
        if len(toks) == 2 and toks[0] in self.colors and toks[1] in self.colors:
            if toks[0] == toks[1]:
                return f"{toks[0].title()}."
            return f"{toks[0].title()} and {toks[1].title()}."

        # Food pairs
        if len(toks) == 2 and toks[0] in self.foods and toks[1] in self.foods:
            if toks[0] == toks[1]:
                return f"{toks[0].title()}."
            return f"{toks[0].title()} and {toks[1].title()}."

        # Default
        return self._to_sentence(" ".join(t.title() for t in toks))

    # ---------------------------
    # helpers
    # ---------------------------
    def _dedupe_consecutive(self, toks: List[str]) -> List[str]:
        out = []
        for t in toks:
            if not out or t != out[-1]:
                out.append(t)
        return out

    def _to_sentence(self, s: str, punct: str = ".") -> str:
        s = s.strip()
        if not s:
            return ""
        if punct and not s.endswith((".", "!", "?", ",")):
            return s + punct
        return s