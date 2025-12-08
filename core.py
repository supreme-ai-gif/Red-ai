# ========================= core.py ================================

import os, json, time, re
import numpy as np

DEFAULT_MEMORY = {
    "name": "Genetic",
    "personality": "curious",
    "user_name": "",
    "learned_facts": {},
    "response_fitness": {},
    "weight": {},            
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


# =================================================================
class GeneticCore:
    def __init__(self, voice):
        self.voice = voice
        self.weight = 1.0  # optional, only if you intend to use a scalar
        self.memory = self._load_memory()
        self._ensure_files()
        self.weights = self._load_weights()
        self.response_count = []

        # init fitness for every response
        for i in range(len(RESPONSES)):
            self.memory["response_fitness"].setdefault(str(i), 1.0)


# =================================================================
    def _ensure_files(self):
        if not os.path.exists(CONV_FILE):
            with open(CONV_FILE,"w") as f: json.dump([],f)


    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            return json.load(open(MEMORY_FILE))
        
        m = DEFAULT_MEMORY.copy()
        m["meta"]["created_at"] = time.time()
        return m


# ========================= WEIGHTS ==============================
    def _init_weight(self):
        return {
            "W1": np.random.randn(2,12).tolist(),
            "b1": np.random.randn(12).tolist(),
            "W2": np.random.randn(12,len(RESPONSES)).tolist(),
            "b2": np.random.randn(len(RESPONSES)).tolist()
        }


    def _load_weight(self):
        loaded = None
        try:
            if "weight" in self.memory:
                loaded = {k: np.array(v) for k,v in self.memory["weight"].items()}
        except:
            loaded=None

        # If corrupted OR empty → regenerate brain
        if loaded and all(isinstance(v,np.ndarray) for v in loaded.values()):
            return loaded

        brain = self._init_weight()
        self.memory["weight"] = brain
        self._save_memory()
        return {k:np.array(v) for k,v in brain.items()}


    def _save_memory(self):
        self.memory["weight"] = {k:v.tolist() for k,v in self.weight.items()}
        json.dump(self.memory,open(MEMORY_FILE,"w"),indent=2)


# ========================= BEHAVIOUR CONTROL ====================
    def _in_quiet_hours(self):
        start,end = self.memory["settings"]["quiet_hours"]
        hour=time.localtime().tm_hour
        return (start<=hour<end) if start<end else (hour>=start or hour<end)


    def _can_proactively_speak(self):
        if self.memory["settings"]["mode"]!="proactive": return False
        if self.memory["settings"]["muted"]: return False
        if self._in_quiet_hours(): return False

        # hourly speech limiter
        cutoff=time.time()-3600
        self.response_count=[t for t in self.response_count if t>cutoff]

        return len(self.response_count) < self.memory["settings"]["rate_limit_per_hour"]


    def speak(self,text,proactive=False):
        if self.memory["settings"]["muted"]: return
        if proactive and not self._can_proactively_speak(): return

        try: self.voice.speak(text)
        except: print("SPEAK:",text)


# ======================= HISTORY & LEARNING ======================
    def _log_conv(self,u,b,i):
        conv=json.load(open(CONV_FILE))
        conv.append({"time":time.time(),"user":u,"bot":b,"response_index":i})
        json.dump(conv,open(CONV_FILE,"w"),indent=1)


    def _similarity(self,a,b):
        A=set(re.findall(r'\w+',a.lower()))
        B=set(re.findall(r'\w+',b.lower()))
        if not A or not B: return 0
        return len(A&B)/max(len(A),len(B))


    def _find_similar_history(self,q,threshold=0.42):
        try: conv=json.load(open(CONV_FILE))
        except: return None,0.0

        best=None; score=0
        for e in conv:
            s=self._similarity(q,e["user"])
            if s>score: score=s; best=e

        return (best,score) if score>=threshold else (None,0)


# ================= FACT EXTRACTION ====================
    def _extract_fact(self,text):
        tl=text.lower().strip()

        if m:=re.match(r'.*remember (.+):(.+)',tl):
            return m[1].strip(),m[2].strip()
        if m:=re.match(r'.*remember (.+)',tl):
            return m[1].strip(),True
        if m:=re.match(r'.*my (.+) is (.+)',tl):
            return m[1].strip(),m[2].strip()
        if m:=re.match(r'.*(i am|i\'m) (.+)',tl):
            return "user_identity",m[2].strip()

        return None,None


# ======================== PROCESS INPUT ==========================
    def process_input(self,user_input):

        user=user_input.strip()
        if not user: return


        # -----------------------------------------------------------
        # 1. Control Commands
        # -----------------------------------------------------------
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


        # -----------------------------------------------------------
        # 2. Fact Learning
        # -----------------------------------------------------------
        key,val=self._extract_fact(user)

        if key:
            if key=="user_identity":
                self.memory["user_name"]=val
                out=f"Nice meeting you {val}, I will remember you."
                self.speak(out)

            else:
                self.memory["learned_facts"][key]=val
                out=f"I will remember that {key} is {val}"
                self.speak(out)

            self._log_conv(user,out,None)
            self._save_memory()
            return


        # -----------------------------------------------------------
        # 3. Fact Recall
        # -----------------------------------------------------------
        for fk,fv in self.memory["learned_facts"].items():
            if fk.lower() in user.lower():
                out=f"{fk} is {fv}" if fv!=True else f"I remember {fk}."
                self.speak(out)
                self._log_conv(user,out,4)
                self._save_memory()
                return


        # -----------------------------------------------------------
        # 4. History Reuse
        # -----------------------------------------------------------
        best,score=self._find_similar_history(user)
        if best:
            prev=best["bot"]
            self.speak(f"As I said earlier: {prev}")
            self._log_conv(user,"reused:"+prev,best["response_index"])
            self._save_memory()
            return


        # -----------------------------------------------------------
        # 5. Neural Output Response
        # -----------------------------------------------------------
        x=np.array([len(user),len(re.findall(r'\w+',user))],float)
        logits=self._neural_forward(x)

        # weighted by fitness
        combined=[logits[i]*self.memory["response_fitness"].get(str(i),1)
                  for i in range(len(RESPONSES))]

        choice=int(np.argmax(combined))
        bot=RESPONSES[choice]

        # voice output
        self.speak(bot)
        self._log_conv(user,bot,choice)

        self.memory["response_fitness"][str(choice)]+=0.03

        # mutation
        reward=len(user)/50
        for k in self.weight:
            self.weight[k]+=np.random.normal(0,0.02,self.weight[k].shape)*reward
            self.weight[k]/=np.linalg.norm(self.weight[k]) or 1

        self._save_memory()


# =================================================================
    def _neural_forward(self,x):
        W1,b1=self.weight["W1"],self.weight["b1"]
        W2,b2=self.weight["W2"],self.weight["b2"]
        h=np.tanh(x@W1+b1)
        return h@W2+b2
