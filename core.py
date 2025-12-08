core.py
import os, json, time, re
import numpy as np

DEFAULT_MEMORY = {
    "name": "Genetic",
    "personality": "curious",
    "user_name": "",
    "learned_facts": {},
    "response_fitness": {},
    "weights": {},            # persistent neural weights
    "meta": {"created_at": None},
    "settings": {
        "mode": "passive",    # passive / proactive / restricted
        "quiet_hours": [22,7],# do not speak between these hours (22:00-07:00)
        "rate_limit_per_hour": 6,
        "muted": False
    }
}

MEMORY_FILE = "memory.json"
CONV_FILE = "conversations.json"

# Simple canned responses (curious style)
RESPONSES = [
    "Hello! How can I help you today?",                    #0
    "My name is Genetic. What would you like me to learn?",#1
    "That's interesting — tell me more.",                  #2
    "I am still learning. Could you explain that?",       #3
    "I don't know that yet. Please teach me by saying 'remember ...'.", #4
    "Got it. I'll remember that.",                        #5
    "I might be mistaken — please correct me if I'm wrong."#6
]

class GeneticCore:
    def __init__(self, voice):
        self.voice = voice
        self.weight = 1.0
        self.memory = self._load_memory()
        self._ensure_files()
        self.weights = self._load_weights()
        self.response_count = []  # timestamps of proactive utterances for rate limit
        # initialize response fitness
        for i in range(len(RESPONSES)):
            self.memory["response_fitness"].setdefault(str(i), 1.0)

    def _ensure_files(self):
        if not os.path.exists(CONV_FILE):
            with open(CONV_FILE, "w") as f:
                json.dump([], f)

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        else:
            m = DEFAULT_MEMORY.copy()
            m["meta"]["created_at"] = time.time()
            return m

    def _init_weights(self):
        return {
            "W1": np.random.randn(2, 12).tolist(),
            "b1": np.random.randn(12).tolist(),
            "W2": np.random.randn(12, len(RESPONSES)).tolist(),
            "b2": np.random.randn(len(RESPONSES)).tolist()
        }

    
    def _save_memory(self):
        """Save memory and weights to file."""
        self.memory["weights"] = {k: v.tolist() for k, v in self.weights.items()}
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)

    def _load_weights(self):
         """Load neural weights or create default random vectors."""
         loaded = None

    # Attempt to load existing weights from memory
         if "weights" in self.memory and isinstance(self.memory["weights"], dict):
             try:
                 loaded = {k: np.array(v, dtype=float) for k, v in self.memory["weights"].items()}
             except Exception:
                 print("⚠ Weight load failed → resetting")
                 loaded = None

    # Basic corruption guard: make sure all values are numpy arrays
         if loaded and all(isinstance(v, np.ndarray) for v in loaded.values()):
             return loaded

    # If loading failed or missing, initialize a new brain
         w = self._init_weights()
         self.memory["weights"] = w
         self._save_memory()

    # Convert to numpy arrays for internal use
         return {k: np.array(v) for k, v in w.items()} 

    # --- utility: is it quiet hours?
    def _in_quiet_hours(self):
        q = self.memory["settings"].get("quiet_hours", [22,7])
        start, end = q
        now = time.localtime()
        hour = now.tm_hour
        if start < end:
            return start <= hour < end
        else:
            return hour >= start or hour < end

    # --- rate limiter
    def _can_proactively_speak(self):
        mode = self.memory["settings"].get("mode","passive")
        if mode != "proactive":
            return False
        if self.memory["settings"].get("muted", False):
            return False
        if self._in_quiet_hours():
            return False
        # prune timestamps older than 1 hour
        cutoff = time.time() - 3600
        self.response_count = [t for t in self.response_count if t > cutoff]
        return len(self.response_count) < self.memory["settings"].get("rate_limit_per_hour", 6)

    def _record_proactive(self):
        self.response_count.append(time.time())

    # --- speak wrapper with checks
    def speak(self, text, proactive=False):
        if self.memory["settings"].get("muted"):
            return
        if proactive:
            if not self._can_proactively_speak():
                return
            self._record_proactive()
        # proceed to speak
        try:
            self.voice.speak(text)
        except Exception:
            print("SPEAK:", text)

    def wakeup(self):
        name = self.memory.get("name","Genetic")
        intro = f"Hello, I am {name}. I am running and ready to learn."
        self.speak(intro, proactive=False)

    # --- Logging conversation
    def _log_conv(self, user_text, bot_text, response_index):
        entry = {
            "timestamp": time.time(),
            "user": user_text,
            "bot": bot_text,
            "response_index": response_index
        }
        with open(CONV_FILE, "r") as f:
            conv = json.load(f)
        conv.append(entry)
        with open(CONV_FILE, "w") as f:
            json.dump(conv, f, indent=1)

    # --- simple similarity
    def _similarity(self, a, b):
        a_tokens = set(re.findall(r'\w+', a.lower()))
        b_tokens = set(re.findall(r'\w+', b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = a_tokens.intersection(b_tokens)
        return len(inter) / max(len(a_tokens), len(b_tokens))

    def _find_similar_history(self, query, threshold=0.45):
        try:
            with open(CONV_FILE, "r") as f:
                conv = json.load(f)
        except Exception:
            return None, 0.0
        best = None; best_score = 0.0
        for e in conv:
            s = self._similarity(query, e.get("user",""))
            if s > best_score:
                best_score = s; best = e
        if best_score >= threshold:
            return best, best_score
        return None, 0.0

    # --- neural forward & mutate
    def _neural_forward(self, x):
        W1 = self.weights["W1"]; b1 = self.weights["b1"]
        W2 = self.weights["W2"]; b2 = self.weights["b2"]
        h = np.tanh(np.dot(x, W1) + b1)
        out = np.dot(h, W2) + b2
        return out

    def _mutate_weights(self, strength=0.07):
        for k in self.weights:
            self.weights[k] = self.weights[k] + strength * np.random.randn(*self.weights[k].shape)

    # --- extract facts
    def _extract_fact(self, text):
        tl = text.strip().lower()
        m = re.match(r'.*remember\s+(.+):\s*(.+)', tl)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.match(r'.*remember\s+(.+)', tl)
        if m:
            return m.group(1).strip(), True
        m = re.match(r'.*my\s+([a-z0-9_ ]+)\s+is\s+(.+)', tl)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.match(r'.*\b(i am|i\'m)\s+(.+)', tl)
        if m:
            return "user_identity", m.group(2).strip()
        return None, None
    # process_input
    def process_input(self, user_input):
        user = user_input.strip()
        if not user:
            return

        text = user  # for evolution calculations

    # --- control commands
    cmd = user.lower().strip()
    if cmd.startswith("set mode "):
        mode = cmd.split("set mode ",1)[1].strip()
        if mode in ("passive","proactive","restricted"):
            self.memory["settings"]["mode"] = mode
            self.speak(f"Mode set to {mode}.", proactive=False)
            self._save_memory()
            return
    if cmd in ("mute","pause"):
        self.memory["settings"]["muted"] = True
        self.speak("I am muted until you ask me to speak again.", proactive=False)
        self._save_memory()
        return
    if cmd in ("unmute","resume"):
        self.memory["settings"]["muted"] = False
        self.speak("I am unmuted.", proactive=False)
        self._save_memory()
        return

    # --- fact extraction
    key, val = self._extract_fact(user)
    if key:
        if key == "user_identity":
            self.memory["user_name"] = val
            self.memory["learned_facts"]["user_name"] = val
            out = f"Nice to meet you, {val}. I will remember that."
            self.speak(out)
            self._log_conv(user, out, None)
            self._save_memory()
            return
        self.memory["learned_facts"][key] = val
        out = f"Okay, I will remember that {key} is {val}" if val is not True else f"Okay, I will remember: {key}"
        self.speak(out)
        # reward learning response (index 5)
        self.memory["response_fitness"]["5"] = self.memory["response_fitness"].get("5",1.0) + 0.1
        self._log_conv(user, out, 5)
        self._save_memory()
        return

    # --- direct fact query
    for fk, fv in self.memory["learned_facts"].items():
        if fk in user.lower():
            out = f"{fk} is {fv}" if fv is not True else f"I remember that {fk}."
            self.speak(out)
            # reward factual response slot
            self.memory["response_fitness"]["4"] = self.memory["response_fitness"].get("4",1.0) + 0.1
            self._log_conv(user, out, 4)
            self._save_memory()
            return

    # --- history reuse
    best_entry, score = self._find_similar_history(user)
    if best_entry:
        prev = best_entry.get("bot","")
        self.speak(f"As I said before: {prev}")
        idx = best_entry.get("response_index")
        if idx is not None:
            self.memory["response_fitness"][str(idx)] = self.memory["response_fitness"].get(str(idx),1.0) + 0.05
        self._log_conv(user, "reused:"+prev, idx)
        self._save_memory()
        return

    # --- Neural decision
    x = np.array([len(user), len(re.findall(r'\w+', user))], dtype=float)
    out = self._neural_forward(x)
    combined = []
    for i, val in enumerate(out):
        fitness = self.memory["response_fitness"].get(str(i), 1.0)
        combined.append(val * fitness)
    choice = int(np.argmax(combined))
    bot_text = RESPONSES[choice]

    # personality tweak
    if self.memory.get("personality","") == "curious":
        if choice == 0:
            bot_text += " I'm curious — what would you like me to learn?"
        elif choice in (2,3):
            bot_text += " Can you give an example?"
        elif choice == 4:
            bot_text = "I don't know that yet. Would you teach me by saying 'remember ...'?"
        elif choice == 6:
            bot_text += " What would be a better answer?"

    # speak
    self.speak(bot_text, proactive=False)
    self._log_conv(user, bot_text, choice)

    # --- simple reinforcement
    self.memory["response_fitness"][str(choice)] = self.memory["response_fitness"].get(str(choice),1.0) + 0.03

    # --- Evolutionary learning step
    mutation_rate = 0.02
    reward = len(text)/50  # longer messages = stronger adaptation

    for key in self.weights:
        noise = np.random.normal(0, mutation_rate, self.weights[key].shape)
        self.weights[key] += noise * reward

    # Normalize to keep brain stable
    for key in self.weights:
        norm = np.linalg.norm(self.weights[key])
        if norm > 0:
            self.weights[key] /= norm

    # --- save memory
    self._save_memory()

        
