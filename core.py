import os, json, time, re
import numpy as np

DEFAULT_MEMORY = {
    "name": "Genetic",
    "personality": "curious",
    "user_name": "",
    "learned_facts": {},
    "response_fitness": {},
    "weights": {},            
    "meta": {"created_at": None},
    "settings": {
        "mode": "passive",   
        "quiet_hours": [22,7],
        "rate_limit_per_hour": 6,
        "muted": False
    }
}

MEMORY_FILE = "memory.json"
CONV_FILE = "conversations.json"

RESPONSES = [
    "Hello! How can I help you today?",
    "My name is Genetic. What would you like me to learn?",
    "That's interesting — tell me more.",
    "I am still learning. Could you explain that?",
    "I don't know that yet. Please teach me by saying 'remember ...'.",
    "Got it. I'll remember that.",
    "I might be mistaken — please correct me if I'm wrong."
]

class GeneticCore:
    def __init__(self, voice):
        self.voice = voice
        self.memory = self._load_memory()
        self._ensure_files()
        self.weights = self._load_weights()
        self.response_count = []

        # init fitness for every response
        for i in range(len(RESPONSES)):
            self.memory["response_fitness"].setdefault(str(i), 1.0)

    # ---------------- FILE UTILITIES ----------------
    def _ensure_files(self):
        if not os.path.exists(CONV_FILE):
            with open(CONV_FILE,"w") as f: json.dump([],f)

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE,"r") as f:
                return json.load(f)
        m = DEFAULT_MEMORY.copy()
        m["meta"]["created_at"] = time.time()
        return m

    def _save_memory(self):
        self.memory["weights"] = {k:v.tolist() for k,v in self.weights.items()}
        with open(MEMORY_FILE,"w") as f:
            json.dump(self.memory,f,indent=2)

    # ---------------- WEIGHTS ----------------
    def _init_weights(self):
        return {
            "W1": np.random.randn(2,12),
            "b1": np.random.randn(12),
            "W2": np.random.randn(12,len(RESPONSES)),
            "b2": np.random.randn(len(RESPONSES))
        }

    def _load_weights(self):
        loaded = None
        try:
            if "weights" in self.memory and isinstance(self.memory["weights"], dict):
                loaded = {k: np.array(v, dtype=float) for k,v in self.memory["weights"].items()}
        except Exception:
            loaded = None

        # validate
        if loaded and all(isinstance(v, np.ndarray) for v in loaded.values()):
            return loaded

        # if missing or corrupted, init new brain
        w = self._init_weights()
        self.memory["weights"] = {k:v.tolist() for k,v in w.items()}
        self._save_memory()
        return w

    # ---------------- BEHAVIOR CONTROL ----------------
    def _in_quiet_hours(self):
        start,end = self.memory["settings"]["quiet_hours"]
        hour=time.localtime().tm_hour
        return (start<=hour<end) if start<end else (hour>=start or hour<end)

    def _can_proactively_speak(self):
        if self.memory["settings"]["mode"]!="proactive": return False
        if self.memory["settings"]["muted"]: return False
        if self._in_quiet_hours(): return False
        cutoff=time.time()-3600
        self.response_count=[t for t in self.response_count if t>cutoff]
        return len(self.response_count) < self.memory["settings"]["rate_limit_per_hour"]

    def _record_proactive(self):
        self.response_count.append(time.time())

    def speak(self, text, proactive=False):
        if self.memory["settings"]["muted"]: return
        if proactive and not self._can_proactively_speak(): return
        self._record_proactive() if proactive else None
        try: self.voice.speak(text)
        except: print("SPEAK:", text)

    # ---------------- LOGGING ----------------
    def _log_conv(self, user_text, bot_text, response_index):
        conv=[]
        try:
            with open(CONV_FILE,"r") as f: conv=json.load(f)
        except: pass
        conv.append({"time":time.time(),"user":user_text,"bot":bot_text,"response_index":response_index})
        with open(CONV_FILE,"w") as f: json.dump(conv,f,indent=1)

    # ---------------- SIMILARITY & HISTORY ----------------
    def _similarity(self,a,b):
        A=set(re.findall(r'\w+',a.lower()))
        B=set(re.findall(r'\w+',b.lower()))
        if not A or not B: return 0
        return len(A&B)/max(len(A),len(B))

    def _find_similar_history(self,q,threshold=0.42):
        try:
            with open(CONV_FILE,"r") as f: conv=json.load(f)
        except: return None,0
        best,score=None,0
        for e in conv:
            s=self._similarity(q,e.get("user",""))
            if s>score: score=s; best=e
        return (best,score) if score>=threshold else (None,0)

    # ---------------- FACT EXTRACTION ----------------
    def _extract_fact(self,text):
        tl=text.lower().strip()
        if m:=re.match(r'.*remember (.+):(.+)',tl): return m[1].strip(),m[2].strip()
        if m:=re.match(r'.*remember (.+)',tl): return m[1].strip(),True
        if m:=re.match(r'.*my (.+) is (.+)',tl): return m[1].strip(),m[2].strip()
        if m:=re.match(r'.*(i am|i\'m) (.+)',tl): return "user_identity",m[2].strip()
        return None,None

    # ---------------- NEURAL ----------------
    def _neural_forward(self,x):
        W1,b1=self.weights["W1"],self.weights["b1"]
        W2,b2=self.weights["W2"],self.weights["b2"]
        h=np.tanh(x@W1+b1)
        return h@W2+b2

    # ---------------- PROCESS INPUT ----------------
    def process_input(self,user_input):
        user=user_input.strip()
        if not user: return

        # --- control commands ---
        cmd=user.lower()
        if cmd.startswith("set mode "):
            m=cmd.split("set mode ",1)[1].strip()
            if m in("passive","proactive","restricted"):
                self.memory["settings"]["mode"]=m
                self.speak(f"Mode set to {m}")
                self._save_memory()
        elif cmd in("mute","pause"):
            self.memory["settings"]["muted"]=True
            self.speak("Voice disabled.")
            self._save_memory()
        elif cmd in("unmute","resume"):
            self.memory["settings"]["muted"]=False
            self.speak("Voice enabled.")
            self._save_memory()

        # --- fact learning ---
        key,val=self._extract_fact(user)
        if key:
            if key=="user_identity":
                self.memory["user_name"]=val
                out=f"Nice meeting you {val}, I will remember you."
            else:
                self.memory["learned_facts"][key]=val
                out=f"I will remember that {key} is {val}"
            self.speak(out)
            self._log_conv(user,out,None)
            self._save_memory()
            return

        # --- fact recall ---
        for fk,fv in self.memory["learned_facts"].items():
            if fk.lower() in user.lower():
                out=f"{fk} is {fv}" if fv!=True else f"I remember {fk}."
                self.speak(out)
                self._log_conv(user,out,4)
                self._save_memory()
                return

        # --- history reuse ---
        best,score=self._find_similar_history(user)
        if best:
            prev=best["bot"]
            self.speak(f"As I said earlier: {prev}")
            self._log_conv(user,"reused:"+prev,best["response_index"])
            self._save_memory()
            return

        # --- neural response ---
        x=np.array([len(user),len(re.findall(r'\w+',user))],float)
        logits=self._neural_forward(x)
        combined=[logits[i]*self.memory["response_fitness"].get(str(i),1) for i in range(len(RESPONSES))]
        choice=int(np.argmax(combined))
        bot=RESPONSES[choice]
        self.speak(bot)
        self._log_conv(user,bot,choice)
        self.memory["response_fitness"][str(choice)]+=0.03

        # mutation
        reward=len(user)/50
        for k in self.weights:
            self.weights[k]+=np.random.normal(0,0.02,self.weights[k].shape)*reward
            self.weights[k]/=np.linalg.norm(self.weights[k]) or 1

        self._save_memory()
