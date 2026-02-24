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

        # sets for expansion rules
        self.colors = {
            "BLUE", "RED", "WHITE", "YELLOW", "ORANGE", "PINK", "VIOLET"
        }
        self.foods = {"BREAD", "EGG", "RICE", "LONGANISA"}
        self.days = {"MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"}
        self.time_words = {"TODAY", "TOMORROW"}

        self.greetings = {"GOOD MORNING","GOOD AFTERNOON","GOOD EVENING","THANK YOU","YOURE WELCOME"}

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

        # safety cap
        if len(self.tokens) >= self.max_tokens:
            return self.finalize()

        return None

    def update_pause(self, hands_detected: bool) -> Optional[Tuple[str, str]]:
        now = time.time()

        if hands_detected:
            self.pause_start_time = None
            return None

        # hands not detected
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
    # Expansion (gloss -> English)
    # ---------------------------
    def expand(self, raw: str) -> str:
        toks = [t for t in raw.split() if t]

        if not toks:
            return ""

        # 1) Remove consecutive duplicates (BLUE BLUE -> BLUE)
        toks = self._dedupe_consecutive(toks)

        # 2) If entire thing is one of the multi-word greetings, keep it clean
        joined = " ".join(toks)
        if joined in self.greetings:
            return self._to_sentence(joined.title(), punct="")

        # 3) TODAY/TOMORROW + weekday -> Today is Monday.
        if len(toks) == 2 and toks[0] in self.time_words and toks[1] in self.days:
            return f"{toks[0].title()} is {toks[1].title()}."

        # 4) THANK YOU <-> YOURE WELCOME as two-part exchange
        if joined == "THANK YOU YOURE WELCOME":
            return "Thank you. You're welcome."
        if joined == "YOURE WELCOME THANK YOU":
            return "You're welcome. Thank you."

        # 5) I/YOU + DEAF -> I am deaf / You are deaf
        if len(toks) == 2 and toks[0] in {"I","YOU"} and toks[1] == "DEAF":
            subj = "I" if toks[0] == "I" else "You"
            verb = "am" if subj == "I" else "are"
            return f"{subj} {verb} deaf."

        # 6) I WRONG / WRONG I / YOU WRONG / WRONG YOU -> (swap if needed)
        if len(toks) == 2 and set(toks) & {"I", "YOU"} and "WRONG" in toks:
            subj = "I" if "I" in toks else "You"
            verb = "am" if subj == "I" else "are"
            return f"{subj} {verb} wrong."

        # 7) YES CORRECT / NO WRONG
        if joined == "YES CORRECT":
            return "Yes, correct."
        if joined == "NO WRONG":
            return "No, wrong."

        # 8) I KNOW / YOU KNOW / I UNDERSTAND / YOU UNDERSTAND / I DON'T KNOW ...
        if len(toks) == 2 and toks[0] in {"I","YOU"} and toks[1] in {"KNOW","UNDERSTAND","DON'T","DON'T KNOW"}:
            # handle "DON'T KNOW" possibly split incorrectly
            if toks[1] == "DON'T":
                # if the raw had "DON'T KNOW" it would be 3 tokens; handle below
                pass

        if len(toks) == 3 and toks[0] in {"I","YOU"} and toks[1] == "DON'T" and toks[2] == "KNOW":
            subj = "I" if toks[0] == "I" else "You"
            return f"{subj} don't know."

        if len(toks) == 2 and toks[0] in {"I","YOU"} and toks[1] in {"KNOW","UNDERSTAND"}:
            subj = "I" if toks[0] == "I" else "You"
            return f"{subj} {toks[1].lower()}."

        # 9) Color pairs -> "Blue and red."
        if len(toks) == 2 and toks[0] in self.colors and toks[1] in self.colors:
            if toks[0] == toks[1]:
                return f"{toks[0].title()}."
            return f"{toks[0].title()} and {toks[1].title()}."

        # 10) Food pairs -> "Rice and egg."
        if len(toks) == 2 and toks[0] in self.foods and toks[1] in self.foods:
            if toks[0] == toks[1]:
                return f"{toks[0].title()}."
            return f"{toks[0].title()} and {toks[1].title()}."

        # 11) I YOU / YOU I -> "You and me."
        if len(toks) == 2 and set(toks) == {"I","YOU"}:
            # natural English tends to say "You and me."
            return "You and me."

        # 12) HOT COFFEE / COLD COFFEE
        if len(toks) == 2 and toks[1] == "COFFEE" and toks[0] in {"HOT","COLD"}:
            return f"{toks[0].title()} coffee."

        # Default: just title-case, keep gloss order
        return self._to_sentence(" ".join([t.title() for t in toks]))

    # ---------------------------
    # helpers
    # ---------------------------
    def _dedupe_consecutive(self, toks: List[str]) -> List[str]:
        if not toks:
            return toks
        out = [toks[0]]
        for t in toks[1:]:
            if t != out[-1]:
                out.append(t)
        return out

    def _to_sentence(self, s: str, punct: str = ".") -> str:
        s = s.strip()
        if not s:
            return ""
        if punct and not s.endswith((".", "!", "?", ",")):
            return s + punct
        return s