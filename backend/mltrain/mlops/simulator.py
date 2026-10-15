"""Demo traffic: synthetic loan applications sent to a deployment, with drift scenarios and delayed labels.

    python -m mltrain.mlops.simulator --url http://127.0.0.1:8000 --deployment loan-default --key KEY \
        --scenario downturn --rps 4 --ramp-minutes 20 --label-delay 30

The drift intensity ramps from 0 to 1 over --ramp-minutes (sudden scenarios jump at 1/3 of the ramp).
Ground-truth outcomes (did the applicant default?) are reported --label-delay seconds after each prediction,
for --label-fraction of requests — like loan outcomes that only become known later.
"""
from __future__ import annotations

import argparse
import heapq
import math
import signal
import sys
import time

import numpy as np

from ..sdk.mltrain_client import Client
from .demo import drift_params, generate

SUDDEN = {"new_segment", "broken_feature"}


def intensity(scenario: str, elapsed_min: float, ramp: float) -> float:
    if scenario == "none":
        return 0.0
    if scenario in SUDDEN:
        return 1.0 if elapsed_min >= ramp / 3 else 0.0
    return min(1.0, elapsed_min / max(ramp, 1e-9))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--scenario", default="none")
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--ramp-minutes", type=float, default=20.0)
    ap.add_argument("--label-delay", type=float, default=30.0)
    ap.add_argument("--label-fraction", type=float, default=0.8)
    ap.add_argument("--duration-minutes", type=float, default=0, help="0 = run until stopped")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)

    client = Client(a.url, a.deployment, a.key)
    rng = np.random.default_rng(a.seed)
    pending: list = []                       # (due_time, request_id, label)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    t0, sent, last_report = time.time(), 0, 0.0
    print(f"simulating '{a.scenario}' traffic to {a.deployment} at {a.rps} req/s", flush=True)
    while not stop["flag"]:
        now = time.time()
        elapsed = (now - t0) / 60
        if a.duration_minutes and elapsed >= a.duration_minutes:
            break
        k = rng.poisson(a.rps)                                   # requests this second
        if k:
            t = intensity(a.scenario, elapsed, a.ramp_minutes)
            df, y = generate(k, rng, drift_params(a.scenario, t))
            records = [{key: (None if isinstance(v, float) and math.isnan(v) else v) for key, v in r.items()}
                       for r in df.to_dict("records")]
            try:
                out = client.predict(records)
                for rid, label in zip(out["request_ids"], y):
                    if rng.random() < a.label_fraction:
                        heapq.heappush(pending, (now + a.label_delay * rng.uniform(0.5, 1.5), rid, "default" if label else "repaid"))
                sent += k
            except Exception as e:
                print(f"request failed: {e}", flush=True)
        while pending and pending[0][0] <= now:
            _, rid, label = heapq.heappop(pending)
            client.feedback(rid, label)
        if now - last_report > 30:
            print(f"t={elapsed:.1f}min intensity={intensity(a.scenario, elapsed, a.ramp_minutes):.2f} sent={sent} "
                  f"pending_labels={len(pending)}", flush=True)
            last_report = now
        time.sleep(max(0.0, 1.0 - (time.time() - now)))
    client.close()
    print(f"stopped after {sent} requests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
