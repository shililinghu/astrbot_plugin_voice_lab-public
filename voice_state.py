"""Persistent, isolated voice preferences. No network or AstrBot dependency."""
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class VoiceStore:
    def __init__(self, folder, base_voice="zh-CN-XiaoxiaoNeural"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.base_voice = base_voice

    def path(self, key):
        return self.folder / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def load(self, key):
        path = self.path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"current": {"voice": self.base_voice, "rate": 0, "pitch": 0},
                "candidate": None, "previous": None, "history": []}

    def save(self, key, state):
        fd, name = tempfile.mkstemp(dir=self.folder, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path(key))
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def candidate(state, rate_delta, pitch_delta):
        for value, limit in ((rate_delta, 5), (pitch_delta, 5)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("参数必须是整数")
            if not -limit <= value <= limit or value != int(value):
                raise ValueError("每次只能在 -5 到 +5 之间调整整数值")
        if rate_delta and pitch_delta:
            raise ValueError("每次只调整语速或音调中的一项")
        result = copy.deepcopy(state["current"])
        result["rate"] += int(rate_delta)
        result["pitch"] += int(pitch_delta)
        if not -25 <= result["rate"] <= 25 or not -30 <= result["pitch"] <= 30:
            raise ValueError("达到调音范围：语速 ±25%，音调 ±30Hz")
        return result

    @staticmethod
    def remember(state, action, feedback):
        state["history"].append({"time": datetime.now(timezone.utc).isoformat(),
                                 "action": action, "feedback": feedback[:500]})
        state["history"] = state["history"][-30:]

    def apply(self, key, action, feedback=""):
        state = self.load(key)
        if action == "accept":
            if not state["candidate"]:
                raise ValueError("还没有已生成的候选声音，请先试听")
            state["previous"] = copy.deepcopy(state["current"])
            state["current"] = state["candidate"]
            state["candidate"] = None
        elif action == "reject":
            state["candidate"] = None
        elif action == "undo":
            if state["previous"] is None:
                raise ValueError("没有可回退的版本")
            state["current"], state["previous"] = state["previous"], state["current"]
            state["candidate"] = None
        else:
            raise ValueError("仅支持 accept/reject/undo")
        self.remember(state, action, feedback)
        self.save(key, state)
        return state
