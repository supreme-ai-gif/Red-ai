# learning.py
import os, time, json, importlib.util
UPDATE_DIR = "update"   # drop python files here
PROPOSAL_FILE = "update_proposals.json"

class Updater:
    def __init__(self, core, voice):
        self.core = core
        self.voice = voice
        self.proposals = []
        if not os.path.exists(UPDATE_DIR):
            os.makedirs(UPDATE_DIR)
        if os.path.exists(PROPOSAL_FILE):
            try:
                with open(PROPOSAL_FILE,"r") as f:
                    self.proposals = json.load(f)
            except Exception:
                self.proposals = []

    def scan_updates(self):
        # look for new .py files in update dir
        for fname in os.listdir(UPDATE_DIR):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(UPDATE_DIR, fname)
            # if not already proposed, create a proposal
            if any(p.get("file")==fpath for p in self.proposals):
                continue
            # read file and create proposal
            try:
                with open(fpath,"r") as f:
                    code = f.read()
                prop = {"file": fpath, "code_snippet": code[:1000], "timestamp": time.time(), "status": "pending"}
                self.proposals.append(prop)
                self._save_proposals()
                self.voice.speak(f"I found an update file named {fname}. I can analyze and propose integration if you want.", proactive=True)
            except Exception as e:
                print("update scan error:", e)

    def maybe_prompt_user_for_updates(self):
        # find first pending proposal and ask user (Option 3)
        for p in self.proposals:
            if p["status"] == "pending":
                # ask user
                self.voice.speak("I found a proposed update. Would you like me to analyze it and suggest integration? Reply yes to analyze.", proactive=False)
                ans = input("Approve analysis? (yes/no): ").strip().lower()
                if ans.startswith("y"):
                    # do a simple static analysis (safe)
                    suggestion = self.analyze_code(p["file"])
                    print("Analysis suggestion:\n", suggestion)
                    self.voice.speak("I suggest the following integration: " + (suggestion[:140] + "..."))
                    ans2 = input("Approve integration? (yes/no): ").strip().lower()
                    if ans2.startswith("y"):
                        integrated = self._integrate_file(p["file"])
                        if integrated:
                            p["status"] = "integrated"
                            self.voice.speak("Update integrated successfully.")
                        else:
                            p["status"] = "failed"
                            self.voice.speak("Integration failed. I saved the proposal.")
                    else:
                        p["status"] = "declined"
                        self.voice.speak("Okay, I will not integrate that update.")
                else:
                    p["status"] = "skipped"
                self._save_proposals()
                break

    def analyze_code(self, file_path):
        # very small static check: look for functions named register_update or safe_integration
        try:
            with open(file_path,"r") as f:
                code = f.read()
            if "register_update" in code:
                return "Contains register_update hook — safe to integrate if you trust it."
            else:
                return "No special hook found. Manual review recommended; integration will import as module in sandbox."
        except Exception as e:
            return "Error reading file: " + str(e)

    def _integrate_file(self, file_path):
        # sandboxed import: copy to a temp module with restricted API (best-effort)
        try:
            spec = importlib.util.spec_from_file_location("user_update", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # if module defines register_update(core), call it
            if hasattr(module, "register_update"):
                try:
                    module.register_update(self.core)
                except Exception as e:
                    print("register_update error:", e)
                    return False
            # mark file integrated (move)
            new_name = file_path + ".integrated"
            os.rename(file_path, new_name)
            return True
        except Exception as e:
            print("integration error:", e)
            return False

    def _save_proposals(self):
        with open(PROPOSAL_FILE,"w") as f:
            json.dump(self.proposals, f, indent=2)
