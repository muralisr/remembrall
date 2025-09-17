from __future__ import annotations
import time, json, sys
from dataclasses import dataclass

@dataclass
class LogEvent:
    ts: float
    level: str
    msg: str
    data: dict

def log(level: str, msg: str, **data):
    evt = LogEvent(time.time(), level.upper(), msg, data)
    print(json.dumps(evt.__dict__), file=sys.stdout, flush=True)

def info(msg: str, **data):
    log("info", msg, **data)

def warn(msg: str, **data):
    log("warn", msg, **data)

def error(msg: str, **data):
    log("error", msg, **data)
