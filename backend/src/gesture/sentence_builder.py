# backend/src/gesture/sentence_builder.py
"""
SentenceBuilder — updated for 34-label FSL dataset (added NAME).

Labels by category:
  POLITENESS : HELLO, PLEASE, THANKS, SORRY, GOODBYE, MORNING, AFTERNOON
  ACTIONS    : WANT, HELP, GO, EAT, SLEEP, UNDERSTAND, KNOW
  QUESTIONS  : HOW, WHAT, WHERE, WHY, WHO
  PEOPLE     : I, YOU, ME, FRIEND, FAMILY
  ANSWERS    : YES, NO, OKAY, GOOD, BAD
  TIME       : TODAY
  PLACE      : HOME, HERE, FROM
  IDENTITY   : NAME
"""

import time
from typing import List, Optional, Tuple


class SentenceBuilder:

    def __init__(self, short_pause: float = 0.8, long_pause: float = 2.2, max_tokens: int = 25):
        self.short_pause   = short_pause
        self.long_pause    = long_pause
        self.max_tokens    = max_tokens

        self.tokens:           List[str]       = []
        self.pause_start_time: Optional[float] = None
        self.last_token_time:  Optional[float] = None

        # ── Category sets ─────────────────────────────────────────────────────
        self.politeness = {"HELLO", "PLEASE", "THANKS", "SORRY", "GOODBYE", "MORNING", "AFTERNOON"}
        self.actions    = {"WANT", "HELP", "GO", "EAT", "SLEEP", "UNDERSTAND", "KNOW"}
        self.questions  = {"HOW", "WHAT", "WHERE", "WHY", "WHO"}
        self.people     = {"I", "YOU", "ME", "FRIEND", "FAMILY"}
        self.subjects   = {"I", "YOU", "ME"}
        self.answers    = {"YES", "NO", "OKAY", "GOOD", "BAD"}
        self.time_words = {"TODAY"}
        self.places     = {"HOME", "HERE", "FROM"}
        self.identity   = {"NAME"}

        # tokens that should never appear in a sentence
        self.ignore_tokens = {
            "WAITING", "COLLECTING...", "BUFFERING...", "UNKNOWN",
            "WAITING...", "SIGNING...", "TOO SHORT / IGNORED", ""
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Token collection
    # ──────────────────────────────────────────────────────────────────────────
    def add_token(self, token: str) -> Optional[Tuple[str, str]]:
        now   = time.time()
        token = token.strip().upper()

        if token in self.ignore_tokens:
            return None

        # skip immediate consecutive duplicates
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

        elapsed = now - self.pause_start_time

        # Short pause (0.8s) with 3+ tokens — likely a complete phrase
        if elapsed >= self.short_pause and len(self.tokens) >= 3:
            return self.finalize()

        # Long pause (2.2s) — finalize anything, even single words
        if elapsed >= self.long_pause and self.tokens:
            return self.finalize()

        return None

    def finalize(self) -> Tuple[str, str]:
        raw = " ".join(self.tokens).strip()
        eng = self.expand(raw)
        self.tokens           = []
        self.pause_start_time = None
        self.last_token_time  = None
        return raw, eng

    def reset(self):
        self.tokens           = []
        self.pause_start_time = None
        self.last_token_time  = None

    # ──────────────────────────────────────────────────────────────────────────
    # Canonicalization — fix token order before expansion
    # ──────────────────────────────────────────────────────────────────────────
    def _canonicalize(self, toks: List[str]) -> List[str]:
        toks = self._dedupe_consecutive(toks)

        # PLEASE always comes first (unless question word present)
        has_question = any(t in self.questions for t in toks)
        if "PLEASE" in toks and toks[0] != "PLEASE" and not has_question:
            toks.remove("PLEASE")
            toks.insert(0, "PLEASE")

        # Question word comes first
        for q in self.questions:
            if q in toks and toks[0] != q:
                toks.remove(q)
                toks.insert(0, q)
                break

        # WHERE + subject + GO/HOME → normalize
        if "WHERE" in toks and any(p in toks for p in self.people):
            return ["WHERE"] + [t for t in toks if t != "WHERE"]

        # Subject–Verb–Object ordering
        subj  = next((t for t in toks if t in self.subjects), None)
        verb  = next((t for t in toks if t in self.actions),  None)

        if subj and verb:
            rest = [t for t in toks if t not in {subj, verb}]
            toks = [subj, verb] + rest

        return toks

    # ──────────────────────────────────────────────────────────────────────────
    # Expansion — gloss tokens → natural English
    # ──────────────────────────────────────────────────────────────────────────
    def expand(self, raw: str) -> str:
        toks = [t for t in raw.upper().split() if t]
        if not toks:
            return ""

        # Check explicit multi-word patterns BEFORE canonicalization
        # so token reordering doesn't break exact-match lookups
        pre = " ".join(toks)

        # ── Pre-canonicalization exact matches ────────────────────────────────
        # "I don't understand" variants
        if pre in {"I NO UNDERSTAND", "ME NO UNDERSTAND",
                   "I NOT UNDERSTAND", "ME NOT UNDERSTAND"}:
            return "I don't understand."

        # "Please help me" variants
        if pre in {"PLEASE HELP ME", "PLEASE HELP I"}:
            return "Please help me."
        if pre in {"HELP ME PLEASE", "HELP I PLEASE"}:
            return "Help me, please!"

        # "I go home" / "You go home"
        if pre in {"I GO HOME", "ME GO HOME"}:
            return "I'm going home."
        if pre == "YOU GO HOME":
            return "You're going home."

        # "You go home today"
        if pre == "YOU GO HOME TODAY":
            return "You go home today."
        if pre == "GO HOME TODAY":
            return "Going home today."

        # "I help you"
        if pre == "I HELP YOU":
            return "I will help you."

        # Greetings with people names (before canonicalizer)
        if pre == "HELLO FRIEND":           return "Hello, friend."
        if pre == "HELLO FAMILY":           return "Hello, family."
        if pre == "GOODBYE FRIEND":         return "Goodbye, friend."
        if pre == "GOODBYE FAMILY":         return "Goodbye, family."
        if pre == "MORNING FRIEND":         return "Good morning, friend."
        if pre == "MORNING FAMILY":         return "Good morning, family."
        if pre == "AFTERNOON FRIEND":       return "Good afternoon, friend."
        if pre == "AFTERNOON FAMILY":       return "Good afternoon, family."

        # "I need help" / "I want help" → always "I need help"
        if pre in {"I WANT HELP", "ME WANT HELP", "I NEED HELP", "ME NEED HELP"}:
            return "I need help."
        if pre in {"YOU WANT HELP", "YOU NEED HELP"}:
            return "You need help."

        # "What do you want?" variants (before canonicalizer reorders YOU)
        if pre in {"WHAT WANT YOU", "WHAT YOU WANT", "YOU WANT WHAT"}:
            return "What do you want?"

        # "I want to go home" variants
        if pre in {"I WANT GO HOME", "ME WANT GO HOME"}:
            return "I want to go home."
        if pre == "YOU WANT GO HOME":
            return "You want to go home."

        # "Hello, how are you?"
        if pre in {"HELLO HOW YOU", "HELLO HOW ARE YOU"}:
            return "Hello! How are you?"

        # "Okay, where?"
        if pre in {"OKAY WHERE", "OKAY WHERE YOU"}:
            return "Okay, where?"

        # "I know! I'll help you."
        if pre in {"I KNOW I HELP YOU", "KNOW I HELP YOU"}:
            return "I know! I will help you."

        # "No problem"
        if pre == "NO PROBLEM":
            return "No problem."

        # Common question phrases — must be checked BEFORE the question-word
        # redirect below, because they need exact phrasing preserved
        if pre == "WHERE YOU GO":           return "Where are you going?"
        if pre == "WHERE YOU FROM":         return "Where are you from?"
        if pre == "WHERE YOU HOME":         return "Where is your home?"
        if pre == "WHERE YOU":              return "Where are you?"
        if pre == "WHY YOU HERE":           return "Why are you here?"
        if pre == "WHY YOU GO":             return "Why are you going?"
        if pre == "WHY YOU SLEEP":          return "Why are you sleeping?"
        if pre == "WHO YOU":                return "Who are you?"
        if pre == "WHO FRIEND":             return "Who is your friend?"
        if pre == "WHO FAMILY":             return "Who is your family?"
        if pre == "HOW YOU":                return "How are you?"
        if pre == "HOW I":                  return "How am I?"
        if pre == "HOW ME":                 return "How am I?"

        # If any question word is present anywhere in the token list,
        # route directly to semantic reorder which handles it correctly
        if any(t in self.questions for t in toks) and len(toks) > 1:
            # Only bypass to semantic if it's NOT already an exact-match above
            # (exact matches above already returned, so we're safe here)
            return self._semantic_reorder(toks)

        # Now apply canonicalization for everything else
        toks   = self._canonicalize(toks)
        joined = " ".join(toks)

        # ── Single tokens ─────────────────────────────────────────────────────
        if len(toks) == 1:
            t = toks[0]
            if t == "HELLO":        return "Hello!"
            if t == "GOODBYE":      return "Goodbye!"
            if t == "THANKS":       return "Thank you."
            if t == "SORRY":        return "Sorry."
            if t == "PLEASE":       return "Please."
            if t == "MORNING":      return "Good morning!"
            if t == "AFTERNOON":    return "Good afternoon!"
            if t == "YES":          return "Yes."
            if t == "NO":           return "No."
            if t == "OKAY":         return "Okay."
            if t == "GOOD":         return "Good."
            if t == "BAD":          return "Bad."
            if t == "TODAY":        return "Today."
            if t == "HOME":         return "Home."
            if t == "HERE":         return "Here."
            if t == "FROM":         return "From."
            if t == "HELP":         return "Help!"
            if t == "WANT":         return "Want."
            if t == "GO":           return "Go."
            if t == "EAT":          return "Eat."
            if t == "SLEEP":        return "Sleep."
            if t == "UNDERSTAND":   return "I understand."
            if t == "KNOW":         return "I know."
            if t == "I":            return "I."
            if t == "YOU":          return "You."
            if t == "ME":           return "Me."
            if t == "FRIEND":       return "Friend."
            if t == "FAMILY":       return "Family."
            if t == "NAME":         return "Name."
            if t in self.questions: return f"{t.title()}?"
            return f"{t.title()}."

        # ── Greetings ─────────────────────────────────────────────────────────
        if joined == "HELLO PLEASE":        return "Hello, please."
        if joined == "HELLO THANKS":        return "Hello, thank you."
        if joined == "HELLO GOODBYE":       return "Hello and goodbye."
        if joined == "MORNING HELLO":       return "Good morning!"
        if joined == "AFTERNOON HELLO":     return "Good afternoon!"
        if joined == "SORRY PLEASE":        return "Sorry, please."
        if joined == "THANKS PLEASE":       return "Thank you, please."

        # ── Script: conversation openers ──────────────────────────────────────
        # "Hello!" / "Hello, how are you?"
        if joined == "HELLO HOW YOU":       return "Hello! How are you?"
        if joined == "HELLO YOU":           return "Hello, you!"
        if joined == "GOOD THANKS":         return "Good, thanks."
        if joined == "GOOD THANK YOU":      return "Good, thank you."

        # ── NAME patterns ─────────────────────────────────────────────────────
        if joined == "WHAT NAME YOU":       return "What is your name?"
        if joined == "WHAT YOU NAME":       return "What is your name?"
        if joined == "NAME WHAT YOU":       return "What is your name?"
        if joined == "YOU NAME WHAT":       return "What is your name?"
        if joined == "WHAT NAME":           return "What is the name?"
        if joined == "MY NAME":             return "My name."
        if joined == "I NAME":              return "My name."
        if joined == "ME NAME":             return "My name."
        if joined == "NAME I":              return "My name."
        if joined == "NAME ME":             return "My name."
        if joined == "YOU NAME":            return "Your name."
        if joined == "NAME YOU":            return "Your name."
        if joined == "FRIEND NAME":         return "My friend's name."
        if joined == "FAMILY NAME":         return "My family's name."
        if joined == "WHO NAME":            return "Who is that?"
        if joined == "NAME WHO":            return "Who is that?"

        # ── Questions ─────────────────────────────────────────────────────────
        if joined == "HOW YOU":             return "How are you?"
        if joined == "HOW I":               return "How am I?"
        if joined == "HOW ME":              return "How am I?"
        if joined == "WHAT YOU WANT":       return "What do you want?"
        if joined == "WHAT WANT YOU":       return "What do you want?"
        if joined == "WHAT YOU EAT":        return "What do you eat?"
        if joined == "WHAT YOU KNOW":       return "What do you know?"
        if joined == "WHAT YOU NAME":       return "What is your name?"
        if joined == "WHERE YOU GO":        return "Where are you going?"
        if joined == "WHERE YOU FROM":      return "Where are you from?"
        if joined == "WHERE HOME":          return "Where is home?"
        if joined == "WHERE YOU":           return "Where are you?"
        if joined == "WHO YOU":             return "Who are you?"
        if joined == "WHO FRIEND":          return "Who is your friend?"
        if joined == "WHO FAMILY":          return "Who is your family?"
        if joined == "WHY YOU GO":          return "Why are you going?"
        if joined == "WHY YOU SLEEP":       return "Why are you sleeping?"
        if joined == "WHY YOU HERE":        return "Why are you here?"
        if joined == "WHAT YOU":            return "What about you?"
        if joined == "WHERE YOU HOME":      return "Where is your home?"

        # ── Script: "What you want?" ──────────────────────────────────────────
        if joined == "WHAT WANT":           return "What do you want?"

        # ── YES/NO + action ───────────────────────────────────────────────────
        if joined == "YES UNDERSTAND":      return "Yes, I understand."
        if joined == "NO UNDERSTAND":       return "No, I don't understand."
        if joined == "YES KNOW":            return "Yes, I know."
        if joined == "NO KNOW":             return "No, I don't know."
        if joined == "OKAY GOOD":           return "Okay, good."
        if joined == "YES GOOD":            return "Yes, good."
        if joined == "NO BAD":              return "No, bad."
        if joined == "OKAY WHERE":          return "Okay, where?"

        # ── Script: "I not understand / Please help me" ───────────────────────
        if joined == "I NOT UNDERSTAND":        return "I don't understand."
        if joined == "ME NOT UNDERSTAND":       return "I don't understand."
        if joined == "I NO UNDERSTAND":         return "I don't understand."
        if joined == "ME NO UNDERSTAND":        return "I don't understand."
        if joined == "PLEASE HELP ME":          return "Please help me."
        if joined == "PLEASE HELP I":           return "Please help me."
        if joined == "HELP ME PLEASE":          return "Help me, please!"
        if joined == "I HELP YOU":              return "I will help you."
        if joined == "I HELP":                  return "I will help."

        # ── Script: "You good friend" / "I know! I help you." ─────────────────
        if joined == "YOU GOOD FRIEND":         return "You are a good friend."
        if joined == "I KNOW I HELP YOU":       return "I know! I will help you."
        if joined == "KNOW I HELP YOU":         return "I know! I will help you."

        # ── Script: "No problem / You go home today?" ─────────────────────────
        if joined == "NO PROBLEM":              return "No problem."
        if joined == "YOU GO HOME TODAY":       return "You go home today?"
        if joined == "GO HOME TODAY":           return "Going home today?"

        # ── Script: "Yes. Goodbye!" ───────────────────────────────────────────
        if joined == "YES GOODBYE":             return "Yes, goodbye!"
        if joined == "GOODBYE YES":             return "Yes, goodbye!"

        # ── I/YOU + action (subject–verb) ─────────────────────────────────────
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] in self.actions:
            subj = "I" if toks[0] in {"I", "ME"} else "You"
            verb = toks[1].lower()
            verb_map = {
                "want":       f"{subj} want.",
                "help":       f"Help me!" if subj == "I" else f"{subj} help.",
                "go":         f"{subj} go.",
                "eat":        f"{subj} eat.",
                "sleep":      f"{subj} sleep.",
                "understand": f"{subj} understand.",
                "know":       f"{subj} know.",
            }
            return verb_map.get(verb, f"{subj} {verb}.")

        # ── I/YOU + GO + PLACE ────────────────────────────────────────────────
        if len(toks) == 3 and toks[0] in self.subjects and toks[1] == "GO" and toks[2] in self.places:
            subj  = "I" if toks[0] in {"I", "ME"} else "You"
            place = toks[2].lower()
            place_map = {"home": "home", "here": "here", "from": "from here"}
            return f"{subj} go {place_map.get(place, place)}."

        # ── I/YOU + WANT + action/place ───────────────────────────────────────
        if len(toks) == 3 and toks[0] in self.subjects and toks[1] == "WANT":
            subj = "I" if toks[0] in {"I", "ME"} else "You"
            obj  = toks[2]
            if obj in self.actions:
                return f"{subj} want to {obj.lower()}."
            if obj in self.places:
                return f"{subj} want to go {obj.lower()}."
            if obj in self.people:
                return f"{subj} want {obj.lower()}."
            if obj == "NAME":
                return f"{subj} want to know the name."
            return f"{subj} want {obj.title()}."

        # ── I/YOU + GO + HOME (common phrase) ────────────────────────────────
        if toks in [["I", "GO", "HOME"], ["ME", "GO", "HOME"]]:
            return "I'm going home."
        if toks == ["YOU", "GO", "HOME"]:
            return "You're going home."

        # ── PLEASE + action ───────────────────────────────────────────────────
        if len(toks) == 2 and toks[0] == "PLEASE" and toks[1] in self.actions:
            return f"Please {toks[1].lower()}."
        if len(toks) == 2 and toks[0] == "PLEASE" and toks[1] in self.places:
            return f"Please go {toks[1].lower()}."
        if len(toks) == 3 and toks[0] == "PLEASE" and toks[1] in self.actions and toks[2] in self.subjects:
            obj = "me" if toks[2] in {"I", "ME"} else "you"
            return f"Please {toks[1].lower()} {obj}."

        # ── HELP + subject ────────────────────────────────────────────────────
        if len(toks) == 2 and toks[0] == "HELP" and toks[1] in self.subjects:
            subj = "me" if toks[1] in {"I", "ME"} else "you"
            return f"Help {subj}!"
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] == "HELP":
            subj = "I" if toks[0] in {"I", "ME"} else "You"
            return f"{subj} need help!"

        # ── TODAY + action ────────────────────────────────────────────────────
        if len(toks) == 2 and toks[0] == "TODAY" and toks[1] in self.actions:
            return f"Today, {toks[1].lower()}."
        if len(toks) >= 2 and toks[0] in self.subjects and toks[1] in self.actions and "TODAY" in toks:
            subj  = "I" if toks[0] in {"I", "ME"} else "You"
            verb  = toks[1].lower()
            place = next((t for t in toks if t in self.places), None)
            place_str = (" " + {"HOME":"home","HERE":"here","FROM":"from here"}.get(place, place.lower())) if place else ""
            return f"{subj} {verb}{place_str} today."

        # ── FRIEND/FAMILY + action ────────────────────────────────────────────
        if len(toks) == 2 and toks[0] in {"FRIEND", "FAMILY"} and toks[1] in self.actions:
            return f"{toks[0].title()} {toks[1].lower()}s."
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] in {"FRIEND", "FAMILY"}:
            subj = "My" if toks[0] in {"I", "ME"} else "Your"
            return f"{subj} {toks[1].lower()}."

        # ── FROM + PLACE ──────────────────────────────────────────────────────
        if len(toks) == 2 and toks[0] == "FROM" and toks[1] in self.places:
            return f"From {toks[1].lower()}."
        if len(toks) == 2 and toks[0] in self.subjects and toks[1] == "FROM":
            subj = "I" if toks[0] in {"I", "ME"} else "You"
            return f"{subj} am from here." if subj == "I" else f"{subj} are from here."

        # ── NAME + subject / subject + NAME ───────────────────────────────────
        if len(toks) == 2 and "NAME" in toks:
            other = toks[0] if toks[1] == "NAME" else toks[1]
            if other in {"I", "ME"}:        return "My name."
            if other == "YOU":              return "Your name."
            if other == "FRIEND":           return "My friend's name."
            if other == "FAMILY":           return "My family's name."
            if other in self.questions:     return f"{other.title()} is the name?"

        # ── Questions with question word ──────────────────────────────────────
        if toks[0] in self.questions:
            rest = " ".join(t.lower() for t in toks[1:])
            return f"{toks[0].title()} {rest}?"

        # ── Semantic fallback — handles any unknown word order ────────────────
        return self._semantic_reorder(toks)

    # ──────────────────────────────────────────────────────────────────────────
    # Semantic reorder — handles ANY word order (rambled input)
    # Slot order: POLITENESS → NEGATION → SUBJECT → VERB → OBJECT/PLACE → TIME
    # ──────────────────────────────────────────────────────────────────────────
    def _semantic_reorder(self, toks: List[str]) -> str:
        pool = list(toks)

        def take(candidates):
            for c in candidates:
                if c in pool:
                    pool.remove(c)
                    return c
            return None

        def take_all(category):
            found = [t for t in pool if t in category]
            for f in found:
                pool.remove(f)
            return found

        verb_map  = {
            "GO": "go", "EAT": "eat", "SLEEP": "sleep", "WANT": "want",
            "HELP": "help", "UNDERSTAND": "understand", "KNOW": "know",
        }
        place_map = {"HOME": "home", "HERE": "here", "FROM": "from here"}
        polite_map = {
            "SORRY": "Sorry,",  "PLEASE": "please",     "THANKS": "Thank you,",
            "HELLO": "Hello!",  "GOODBYE": "Goodbye!",  "MORNING": "Good morning!",
            "AFTERNOON": "Good afternoon!",
        }

        # ── Slot extraction ────────────────────────────────────────────────────
        polite_words  = take_all(self.politeness)

        # FIX: question word must be extracted BEFORE places so WHERE is
        # treated as a question word, not a place
        question_word = take(list(self.questions))

        negation      = take({"NO", "NOT"})
        subject_tok   = take(["I", "ME", "YOU"])

        # FIX: extract all action tokens first, then assign roles —
        # prevents the same word appearing as both verb and second_action
        all_actions   = [t for t in pool if t in self.actions]
        verb_tok      = None
        second_action = None

        if "WANT" in all_actions:
            verb_tok = "WANT"
            pool.remove("WANT")
            # secondary is any other action (consumed once from pool)
            other_actions = [t for t in pool if t in self.actions]
            if other_actions:
                second_action = other_actions[0]
                pool.remove(second_action)
        elif all_actions:
            verb_tok = all_actions[0]
            pool.remove(verb_tok)
            # consume any remaining action tokens so they don't appear as leftover
            for extra in [t for t in pool if t in self.actions]:
                pool.remove(extra)

        # FIX: WHERE is already taken as question_word — only pull real places
        place_tok     = take(list(self.places - {"FROM"})) or take(["FROM"])
        time_tok      = take(list(self.time_words))
        answer_tok    = take(list(self.answers - {"NO"}))
        identity_tok  = take(list(self.identity))
        people_tok    = take(list(self.people - {"I", "ME", "YOU"}))
        leftover      = [t for t in pool if t not in self.ignore_tokens]

        # Subject resolution
        subj_eng = poss_eng = None
        if subject_tok in {"I", "ME"}:
            subj_eng = "I";   poss_eng = "my"
        elif subject_tok == "YOU":
            subj_eng = "you"; poss_eng = "your"

        # Default subject to "you" for questions when none signed
        if question_word and subj_eng is None and (verb_tok or identity_tok):
            subj_eng = "you"; poss_eng = "your"

        # ── Question sentence ──────────────────────────────────────────────────
        if question_word:
            q = {"HOW":"How","WHAT":"What","WHERE":"Where",
                 "WHY":"Why","WHO":"Who"}.get(question_word, question_word.title())

            if identity_tok:
                return f"{q} is {(poss_eng + ' ') if poss_eng else ''}name?"

            if verb_tok:
                v   = verb_map.get(verb_tok, verb_tok.lower())
                neg = "don't " if negation else ""

                if verb_tok == "WANT":
                    v2 = (" to " + verb_map[second_action]) if second_action else ""
                    p  = (" " + place_map[place_tok]) if place_tok and not second_action else \
                         (" " + place_map[place_tok]) if place_tok else ""
                    s = subj_eng if subj_eng else "you"
                    return f"{q} do {s} {neg}want{v2}{p}?"

                p = (" " + place_map[place_tok]) if place_tok else ""
                s = subj_eng if subj_eng else "you"

                if subj_eng == "I":
                    return f"{q} do I {neg}{v}{p}?"
                return f"{q} do {s} {neg}{v}{p}?"

            # No verb — question about state/identity
            if subj_eng == "you":
                q_be = {"HOW":"How","WHERE":"Where","WHY":"Why","WHO":"Who"}.get(question_word)
                if q_be:
                    return f"{q_be} are you?"
                return f"{q} do you want?"
            if subj_eng == "I":
                return f"{q} am I?"
            if people_tok:
                return f"{q} is {people_tok.lower()}?"
            return f"{q}?"

        # ── Statement sentence ────────────────────────────────────────────────
        parts = []

        opening = [p for p in polite_words if p != "PLEASE"]
        closing = [p for p in polite_words if p == "PLEASE"]

        for op in opening:
            parts.append(polite_map[op])

        if subj_eng:
            parts.append(subj_eng.capitalize() if not parts else subj_eng)

        if verb_tok:
            if negation:
                if verb_tok == "WANT":
                    v2 = (" to " + verb_map[second_action]) if second_action else ""
                    p  = (" " + place_map[place_tok]) if place_tok else ""
                    parts.append(f"don't want{v2}{p}")
                    place_tok = None
                else:
                    parts.append(f"don't {verb_map.get(verb_tok, verb_tok.lower())}")
            elif verb_tok == "WANT":
                if second_action:
                    p = (" " + place_map[place_tok]) if place_tok else ""
                    parts.append(f"want to {verb_map[second_action]}{p}")
                    place_tok = None
                elif place_tok:
                    parts.append("want to go")
                else:
                    parts.append("want")
            elif verb_tok == "GO" and place_tok:
                parts.append(f"go {place_map[place_tok]}")
                place_tok = None
            else:
                parts.append(verb_map.get(verb_tok, verb_tok.lower()))

        if place_tok:
            parts.append(place_map[place_tok])

        if people_tok:
            parts.append(people_tok.lower())

        if identity_tok:
            parts.append(f"{(poss_eng + ' ') if poss_eng else ''}name")

        if answer_tok and not verb_tok:
            parts.append({"YES":"yes","GOOD":"good","BAD":"bad","OKAY":"okay"}.get(
                answer_tok, answer_tok.lower()))

        if time_tok:
            parts.append("today")

        for cp in closing:
            parts.append(polite_map.get(cp, cp.lower()))

        for lw in leftover:
            parts.append(lw.lower())

        if not parts:
            return " ".join(t.title() for t in toks) + "."

        result = " ".join(parts)
        result = result[0].upper() + result[1:]
        if not result.endswith((".", "!", "?")):
            result += "."
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
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