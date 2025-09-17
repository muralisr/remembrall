from __future__ import annotations
import argparse, os
from remembrall.core import RemembrallRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    runner = RemembrallRunner(config_path=args.config)
    out = runner.run(duration_s=args.duration, fps=args.fps)
    print("Summary:", out)

if __name__ == "__main__":
    main()
