# sentence_builder.py
import time

class SentenceBuilder:
    def __init__(self, short_pause=0.8, long_pause=2.2, max_tokens=20):
        self.short_pause = short_pause
        self.long_pause = long_pause
        self.max_tokens = max_tokens

        self.tokens = []
        self.last_token_time = None
        self.pause_start_time = None

    def add_token(self, token: str):
        now = time.time()
        token = token.strip().upper()

        # Ignore non-words
        if token in ("WAITING", "COLLECTING...", "BUFFERING...", "UNKNOWN", ""):
            return None

        # Avoid immediate duplicates (common with stabilizers)
        if self.tokens and self.tokens[-1] == token:
            self.last_token_time = now
            return None

        self.tokens.append(token)
        self.last_token_time = now

        # Safety cap
        if len(self.tokens) > self.max_tokens:
            return self.finalize()

        return None

    def update_pause(self, hands_detected: bool):
        """Call every frame. Returns finalized (raw, expanded) if a LONG pause happened."""
        now = time.time()

        if hands_detected:
            self.pause_start_time = None
            return None

        # hands not detected
        if self.pause_start_time is None:
            self.pause_start_time = now
            return None

        pause_dur = now - self.pause_start_time

        # Long pause = finalize sentence
        if pause_dur >= self.long_pause and self.tokens:
            return self.finalize()

        return None

    def finalize(self):
        raw = " ".join(self.tokens).strip()
        expanded = self.expand(raw)
        self.tokens = []
        self.last_token_time = None
        self.pause_start_time = None
        return raw, expanded

    def expand(self, raw: str) -> str:
        """
        Simple rule-based gloss->English expansion.
        If we can't expand confidently, return raw (signed order).
        """
        toks = raw.split()

        # Day/time pattern: TODAY MONDAY, TOMORROW TUESDAY
        if len(toks) == 2 and toks[0] in ("TODAY", "TOMORROW"):
            return f"{toks[0].title()} is {toks[1].title()}."

        # I KNOW YOU DEAF -> I know you are deaf.
        if toks == ["I", "KNOW", "YOU", "DEAF"]:
            return "I know you are deaf."

        # YOU DEAF -> You are deaf.
        if toks == ["YOU", "DEAF"]:
            return "You are deaf."

        # I DEAF -> I am deaf.
        if toks == ["I", "DEAF"]:
            return "I am deaf."

        # HOT COFFEE / COLD COFFEE
        if len(toks) == 2 and toks[1] == "COFFEE" and toks[0] in ("HOT", "COLD"):
            return f"{toks[0].title()} coffee."

        # Basic single-word punctuation
        if len(toks) == 1:
            w = toks[0]
            # Greetings
            if w in ("GOOD", "MORNING", "AFTERNOON", "EVENING"):
                return w.title()
            if w in ("THANK", "YOU’RE", "WELCOME", "YOURE", "WELCOME"):
                return w.title()
            return w.title() + "."

        # Default: just title-case words, keep signed order
        return raw.title()