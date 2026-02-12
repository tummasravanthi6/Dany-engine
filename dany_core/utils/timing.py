import time

class StageTimer:
    def __init__(self):
        self._store = {}

    def start(self, name: str):
        self._store[name] = {"start": time.perf_counter(), "duration": None}

    def stop(self, name: str):
        end = time.perf_counter()
        self._store[name]["duration"] = end - self._store[name]["start"]

    def summary(self):
        total = sum(v["duration"] for v in self._store.values())
        return {
            "total_time_sec": round(total, 4),
            "stages": {
                k: round(v["duration"], 4)
                for k, v in self._store.items()
            }
        }
