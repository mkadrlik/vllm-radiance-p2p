#!/usr/bin/env python3
"""Dump ppb_* (backoff sweep, no hammer) lines from a phase0 log."""
import json
import sys

for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("JSON0 "):
            r = json.loads(line[6:])
            if r["test"].startswith("ppb_"):
                print(f"{r['test']:32s} p50={r.get('rtt_p50_us'):>9} "
                      f"p90={r.get('rtt_p90_us'):>10} p99={r.get('rtt_p99_us'):>10} "
                      f"max={r.get('rtt_max_us'):>10} ab={r.get('rtt_aborted')} "
                      f"wall={r['wall_us_per_iter']}")
