"""
Flock Watch core v3 -- simulation aligned with the final ANWE report design.

WHAT THIS IS: an illustrative simulation of a post-anomaly decision-support
pipeline for sheep, NOT a validated detector. Every number (noise levels,
effect sizes, thresholds, rewards, costs) is a placeholder. Nothing here is
evidence that the design works on real farms.

Report design implemented here
------------------------------
* Raceway station: eye thermal proxy, gait asymmetry (RGB-depth), cough
  events (directional microphone), walking speed, skin-lesion score.
  Animals are identified by face recognition; low-confidence identity is NOT
  linked to history and NOT used for learning.
* Baseline: per-animal rolling mean/SD of the ambient-adjusted residual over
  the last N valid observations (flagged periods excluded). z-scores feed a
  single Isolation Forest -> score S in (0, 1].
* Thresholds are QUANTILES of the healthy-score distribution (so they keep
  meaning after a refit), not fixed score values.
* Persistence rule: k of n consecutive passes above the alert threshold
  before Tier 2 (soft isolation). Tier 1 = watchlist (never isolation).
* Exposure: E = passage-time term + shared-pen term, summed over FLAGGED
  animals only; it lowers the WATCH threshold only, never triggers Tier 2.
* Adaptive thresholds: a simplified conservative contextual bandit over
  quantile arms (in the spirit of Kazerouni et al., 2017 -- it is NOT an
  implementation of their algorithm and carries NO formal guarantee here).
  Labels are weighted by evidence source; uncertain outcomes and sensor/ID
  faults never train the model. Random audits supply labels for unflagged
  animals.
* Legal framing: a REPORTABLE-pattern suspicion is recorded as a suspicion
  with its own timestamps. Notification is a human action that does NOT wait
  for the pen-side test. A negative on-farm test or farmer observation cannot
  clear a reportable suspicion. Official confirmation remains with the
  reference laboratory. Where "operational alert" becomes "legal suspicion"
  is a configurable placeholder that must be agreed with veterinary
  authorities.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = ["thermal", "gait", "cough", "speed", "lesion"]
FEATURE_LABELS = {
    "thermal": "Eye thermal proxy",
    "gait": "Gait asymmetry",
    "cough": "Cough events",
    "speed": "Walking speed",
    "lesion": "Skin lesion score",
}
FEATURE_UNITS = {"thermal": "°C", "gait": "/100", "cough": "events", "speed": "m/s", "lesion": "/100"}
MIN_SD = {"thermal": 0.08, "gait": 1.5, "cough": 0.15, "speed": 0.04, "lesion": 1.0}

WINDOW_N = 14          # max valid observations in the rolling baseline
MIN_OBS = 5            # minimum observations before an animal is scored
K_PERSIST, N_PERSIST = 2, 3
WARMUP_ROUNDS = 21
MIN_BUFFER = 60        # minimum healthy z-vectors before the forest is fitted
REFIT_EVERY = 7
ID_CONF_MIN = 0.80
P_DROPOUT = 0.03
P_ID_UNCERTAIN = 0.05
P_TRANSIENT_SPIKE = 0.01
TAU_SECONDS = 30.0     # passage-exposure decay constant
TAU_DAYS = 3.0         # shared-pen exposure decay constant
Q_WATCH = 0.95
Q_WATCH_EXPOSURE_DROP = 0.05   # max lowering of the watch quantile at E = 1
RECOVERY_CLEARANCE_DAYS = 4

PATTERNS = {
    "Febrile infectious": {"thermal": 0.45, "speed": 0.20, "cough": 0.15, "lesion": 0.20},
    "Respiratory": {"cough": 0.55, "thermal": 0.25, "speed": 0.20},
    "Lameness / inflammation": {"gait": 0.70, "speed": 0.30},
    "Dermatological infectious": {"lesion": 0.70, "thermal": 0.30},
}
# "Febrile infectious" stands in for the sheep-pox case study.
REPORTABLE_PATTERNS = {"Febrile infectious"}

# Evidence sources and how much a label from each source counts in learning.
EV_FARMER = "Farmer observation"
EV_PENTEST = "Pen-side rapid test (presumptive)"
EV_VET = "Veterinary assessment"
EV_LAB = "Reference laboratory"
EV_AUDIT = "Random audit (clinical exam)"
EVIDENCE_SOURCES = [EV_FARMER, EV_PENTEST, EV_VET, EV_LAB]
SOURCE_WEIGHT = {EV_FARMER: 0.4, EV_PENTEST: 0.7, EV_VET: 1.0, EV_LAB: 1.0, EV_AUDIT: 1.0}

OUT_POS = "Health problem established"
OUT_NEG = "No problem found after assessment"
OUT_UNC = "Still uncertain (keep unresolved)"
OUT_FAULT = "Sensor / identification fault"
OUTCOMES = [OUT_POS, OUT_NEG, OUT_UNC, OUT_FAULT]

REWARDS = {"TP": 10.0, "FP": -5.0, "FN": -15.0, "TN": 0.0}
REWARD_SHIFT = 15.0    # rewards shifted to be non-negative for the conservative rule

SYSTEM_INTENSIVE = "Intensive / semi-intensive (fixed daily raceway)"
SYSTEM_EXTENSIVE = "Extensive pasture (periodic handling rounds)"
GATE_AUTO = "Automatic diversion (reversible)"
GATE_INSPECT = "Inspection request only (no gate action)"
SUSPICION_RULES = [
    "Reportable pattern + Tier 2 persistence",
    "Any Tier 2 alert",
]


def implied_alert_probability(rewards: Dict[str, float] = None) -> float:
    """Posterior illness probability above which alerting beats staying silent,
    given the reward table (TN assumed 0): p > -r_fp / (r_tp - r_fp - r_fn)."""
    r = rewards or REWARDS
    return -r["FP"] / (r["TP"] - r["FP"] - r["FN"])


class SequenceViolation(Exception):
    """A required order of human/legal steps was not followed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FarmContext:
    system_type: str = SYSTEM_INTENSIVE
    gate_mode: str = GATE_AUTO
    gate_paused: bool = False          # human stop switch for the gate
    suspicion_rule: str = SUSPICION_RULES[0]
    audit_fraction: float = 0.03

    @property
    def handling_interval(self) -> int:
        """Rounds (days) between handling events at which sheep pass the station."""
        return 1 if self.system_type == SYSTEM_INTENSIVE else 4


# ---------------------------------------------------------------------------
# Animals and the simulated world
# ---------------------------------------------------------------------------

BREEDS = ["Chios", "Karagouniko"]


@dataclass
class Animal:
    id: str
    breed: str
    pen: str
    base: Dict[str, float]
    window: Dict[str, Deque[float]] = field(default_factory=lambda: {f: deque(maxlen=WINDOW_N) for f in FEATURES})
    recent_exceed: Deque[bool] = field(default_factory=lambda: deque(maxlen=N_PERSIST))
    event: Optional[dict] = None          # simulated ground truth
    status: str = "active"                # active | official
    isolated: bool = False
    isolation_start: Optional[int] = None
    last_valid_round: int = 0
    last_flag_round: Optional[int] = None
    recoveries: int = 0
    sentinel: bool = False


def _make_animals(rng: random.Random, n: int = 10, sentinel_pct: float = 10.0) -> Dict[str, Animal]:
    animals = {}
    n_pens = max(2, round(n / 20))
    n_sent = max(1, math.ceil(n * sentinel_pct / 100.0))
    for i in range(n):
        aid = f"GR-{2210 + i * 7}"
        base = {
            "thermal": rng.gauss(38.9, 0.12),
            "gait": max(1.0, rng.gauss(6, 1.5)),
            "cough": 0.3,
            "speed": rng.gauss(1.2, 0.08),
            "lesion": max(1.0, rng.gauss(4, 1.0)),
        }
        animals[aid] = Animal(
            id=aid, breed=rng.choice(BREEDS), pen=chr(65 + (i * n_pens) // n),
            base=base, sentinel=(i < n_sent),
        )
    return animals


NOISE_SD = {"thermal": 0.12, "gait": 2.5, "cough": 0.2, "speed": 0.06, "lesion": 1.8}
BETA_THERMAL_TRUE = 0.03   # simulated ambient effect on the thermal reading (°C per °C)
T_REF = 22.0


def _ramp(day: int, ramp_days: float) -> float:
    return min(1.0, max(0.0, day / ramp_days))


def _event_effect(event: Optional[dict]) -> Dict[str, float]:
    """Deterministic shift of each feature produced by the event on its current day."""
    if event is None:
        return {f: 0.0 for f in FEATURES}
    d, p = event["day"], event["pattern"]
    scale = 1.0 if event["kind"] == "disease" else 0.45
    if p == "Febrile infectious":
        sig = 1 / (1 + math.exp(-1.6 * (d - 2.5)))
        eff = {"thermal": 1.8 * sig, "speed": -0.25 * _ramp(d, 5), "cough": 0.8 * _ramp(d, 6),
               "lesion": 25.0 * _ramp(d - 3, 6), "gait": 0.0}
    elif p == "Respiratory":
        eff = {"cough": 3.0 * _ramp(d, 4), "thermal": 0.7 * _ramp(d, 4), "speed": -0.12 * _ramp(d, 5),
               "gait": 0.0, "lesion": 0.0}
    elif p == "Lameness / inflammation":
        eff = {"gait": 40.0 * _ramp(d, 4), "speed": -0.20 * _ramp(d, 4), "thermal": 0.0, "cough": 0.0, "lesion": 0.0}
    else:  # Dermatological infectious
        eff = {"lesion": 35.0 * _ramp(d, 5), "thermal": 0.35 * _ramp(d, 5),
               "gait": 0.0, "cough": 0.0, "speed": 0.0}
    eff = {k: v * scale for k, v in eff.items()}
    td = event.get("treated_day")
    if td is not None and d >= td:
        decay = math.exp(-1.1 * (d - td))
        eff = {k: v * decay for k, v in eff.items()}
    return eff


# ---------------------------------------------------------------------------
# Simplified conservative contextual bandit over quantile arms
# ---------------------------------------------------------------------------

class ConservativeThresholdBandit:
    """Chooses ONE alert-quantile arm per round for the whole flock (not per
    animal, so no individual animal is randomly given a laxer threshold).

    Simplified LinUCB with a conservative gate: a non-baseline arm is only
    allowed when the cumulative (weighted, shifted) reward so far, plus the arm's
    pessimistic estimate for this round, is at least (1 - alpha) times the
    estimated cumulative reward of always using the baseline arm. This mirrors the
    idea in Kazerouni et al. (2017) but is NOT their algorithm and gives no
    formal guarantee with censored, weighted labels."""

    ARMS = [0.975, 0.99, 0.995]
    BASELINE = 1

    PRIOR_MEAN = 12.5          # mid-range of the shifted reward scale (0..25)
    MIN_BASELINE_LABELS = 8    # no exploration before the baseline arm has this much evidence
    MIN_ARM_LABELS = 3         # an arm with less evidence is assumed worst-case (reward 0) in the safety check

    def __init__(self, dim: int = 3, alpha: float = 0.2, c: float = 0.5, arms=None):
        self.dim, self.alpha, self.c = dim, alpha, c
        if arms is not None:
            self.ARMS = list(arms)
        prior = np.zeros(dim)
        prior[0] = self.PRIOR_MEAN
        self.A = [np.eye(dim) for _ in self.ARMS]
        self.b = [prior.copy() for _ in self.ARMS]
        self.counts = [0.0 for _ in self.ARMS]
        self.reward_sum = [0.0 for _ in self.ARMS]
        self.cum_policy = 0.0
        self.cum_base_est = 0.0
        self.log: List[dict] = []

    def _theta(self, a):
        return np.linalg.solve(self.A[a], self.b[a])

    def predict(self, a, x):
        return float(self._theta(a) @ x)

    def bonus(self, a, x):
        return self.c * float(math.sqrt(max(0.0, x @ np.linalg.solve(self.A[a], x))))

    def select(self, x: np.ndarray) -> Tuple[int, dict]:
        base = self.BASELINE
        base_n = self.counts[base]
        emp = (self.reward_sum[base] + REWARD_SHIFT * base_n) / base_n if base_n > 0 else 0.0
        rb = max(emp, self.predict(base, x)) if base_n > 0 else 0.0
        slack = self.cum_policy - (1 - self.alpha) * self.cum_base_est
        allowed = [base]
        if self.counts[base] >= self.MIN_BASELINE_LABELS:
            for a in range(len(self.ARMS)):
                if a == base:
                    continue
                if self.counts[a] < self.MIN_ARM_LABELS:
                    lcb = 0.0                      # no evidence: assume the worst shifted reward
                else:
                    lcb = max(0.0, self.predict(a, x) - self.bonus(a, x))
                if slack + lcb - (1 - self.alpha) * rb >= 0:
                    allowed.append(a)
        best = max(allowed, key=lambda a: self.predict(a, x) + self.bonus(a, x))
        info = {"arm": best, "quantile": self.ARMS[best], "allowed": [self.ARMS[a] for a in allowed],
                "slack": round(slack, 2)}
        return best, info

    def update(self, a: int, x: np.ndarray, reward: float, weight: float = 1.0):
        r = reward + REWARD_SHIFT
        base_n = self.counts[self.BASELINE]
        emp = (self.reward_sum[self.BASELINE] + REWARD_SHIFT * base_n) / base_n if base_n > 0 else r
        rb = max(emp, self.predict(self.BASELINE, x)) if base_n > 0 else r
        self.cum_policy += weight * r
        self.cum_base_est += weight * rb
        self.A[a] = self.A[a] + weight * np.outer(x, x)
        self.b[a] = self.b[a] + weight * r * x
        self.counts[a] += weight
        self.reward_sum[a] += weight * reward

    def table(self) -> List[dict]:
        rows = []
        for a, q in enumerate(self.ARMS):
            rows.append({
                "Alert quantile": q,
                "Role": "baseline" if a == self.BASELINE else "candidate",
                "Weighted labels": round(self.counts[a], 2),
                "Mean reward": round(self.reward_sum[a] / self.counts[a], 2) if self.counts[a] else None,
            })
        return rows


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class AlertRecord:
    id: int
    animal_id: str
    round_no: int
    source: str                       # "raceway" | "audit"
    pattern: str
    S: float
    theta_alert: float
    z: Dict[str, float]
    raw: Dict[str, float]
    resid: Dict[str, float]
    arm_idx: int
    context: List[float]
    gate_action: str                  # diverted | inspection requested | found by audit
    suspicion: bool = False
    suspicion_round: Optional[int] = None
    vet_contacted_round: Optional[int] = None
    authority_notified_round: Optional[int] = None
    pen_test: Optional[str] = None    # "positive" | "negative" (presumptive only)
    outcome: Optional[str] = None
    evidence: Optional[str] = None
    notes: str = ""
    resolved: bool = False
    resolved_round: Optional[int] = None
    isolation_rounds: int = 0
    truth_disease: Optional[bool] = None       # simulation ground truth, revealed on demand
    truth_start_round: Optional[int] = None
    truth_pattern: Optional[str] = None
    pass_s: float = 0.0                        # seconds after the start of the passage session
    exposure_contacts: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_reportable(self) -> bool:
        return self.pattern in REPORTABLE_PATTERNS


# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------

class FlockWatch:
    def __init__(self, seed: Optional[int] = 7, context: Optional[FarmContext] = None, n_animals: int = 10,
                 daily_events: float = 0.3, warmup_rounds: int = WARMUP_ROUNDS, sentinel_pct: float = 10.0,
                 arms=None):
        """daily_events = expected number of NEW simulated health events per day for the whole flock
        (placeholder; 0.3 reproduces the earlier 10-animal setting)."""
        self.rng = random.Random(seed)
        self.daily_events = daily_events
        self.warmup_rounds = warmup_rounds
        self.np_seed = seed if seed is not None else 0
        self.context = context or FarmContext()
        self.animals = _make_animals(self.rng, n_animals, sentinel_pct)
        self.buffer: List[np.ndarray] = []
        self.buffer_scores: np.ndarray = np.array([])
        self.calib_X: Optional[np.ndarray] = None
        self.forest: Optional[IsolationForest] = None
        self.forest_round = -999
        self.beta_thermal = 0.0
        self._beta_obs: Dict[str, List[Tuple[float, float]]] = {}
        self._warm_raw: Dict[str, List[Tuple[float, Dict[str, float]]]] = {a: [] for a in self.animals}
        self.bandit = ConservativeThresholdBandit(arms=arms)
        self.alerts: List[AlertRecord] = []
        self.history: Dict[str, List[dict]] = {a: [] for a in self.animals}
        self.round_no = 0
        self.audit_n = 0
        self.audit_pos = 0
        self.audit_log: List[dict] = []
        self.warnings: List[dict] = []
        self.round_log: List[dict] = []
        self.last_arm = self.bandit.BASELINE
        self.last_context = np.array([1.0, 0.0, 0.0])
        self.last_ambient = T_REF
        self.last_tiers: Dict[str, dict] = {}
        self.counters = {"dropouts": 0, "id_uncertain": 0, "transients": 0}
        self.readings: Dict[str, List[dict]] = {a: [] for a in self.animals}
        self.start_mean: Dict[str, Dict[str, float]] = {}
        self.feed: List[dict] = []
        self.warmup(warmup_rounds)

    # ----- setup -----------------------------------------------------------

    def warmup(self, rounds: int = WARMUP_ROUNDS):
        """Initialization phase: simulated healthy days used to build baselines and fit the forest.
        No disease events, no alerts. (In the real design a 5-10% sentinel cohort wearing
        temporary collars would support this phase.)"""
        for _ in range(rounds):
            self.step(ambient=self.rng.uniform(12, 30), warmup=True)
        # estimate the ambient slope from the whole warm-up, then rebuild windows and the healthy buffer
        for aid, series in self._warm_raw.items():
            self._beta_obs[aid] = [(T, raw["thermal"]) for T, raw in series]
        self._fit_beta()
        self.buffer = []
        for aid, series in self._warm_raw.items():
            animal = self.animals[aid]
            resids = [self._residual(raw, T) for T, raw in series]
            for f in FEATURES:
                animal.window[f].clear()
            for resid in resids:
                z = self._zscores(animal, resid)
                if z is not None:
                    self.buffer.append(np.array([z[f] for f in FEATURES]))
                for f in FEATURES:
                    animal.window[f].append(resid[f])
        self._fit_forest(force=True)
        self.start_mean = {aid: {f: float(np.mean([raw[f] for _, raw in series])) for f in FEATURES}
                           for aid, series in self._warm_raw.items() if series}
        self.round_no = 0
        self.history = {a: [] for a in self.animals}
        self.readings = {a: [] for a in self.animals}
        self.feed = []
        self.round_log = []
        self.warnings = []
        self.alerts = []
        for a in self.animals.values():
            a.last_valid_round = 0
            a.recent_exceed.clear()
        self.counters = {"dropouts": 0, "id_uncertain": 0, "transients": 0}
        self._warm_raw = {a: [] for a in self.animals}

    # ----- model pieces ----------------------------------------------------

    def _fit_forest(self, force: bool = False):
        """Fit the Isolation Forest on healthy readings. The alert/watch thresholds are quantiles of a FROZEN
        held-out healthy reference set (split off once, after warm-up) scored by the current forest, so a
        threshold keeps the same meaning (a fixed false-alert rate on healthy data) after every refit."""
        if len(self.buffer) < MIN_BUFFER and self.calib_X is None:
            return
        if not force and self.round_no - self.forest_round < REFIT_EVERY:
            return
        if self.calib_X is None:
            X = np.vstack(self.buffer)
            idx = np.random.RandomState(self.np_seed).permutation(len(X))
            n_fit = int(0.7 * len(X))
            self.calib_X = X[idx[n_fit:]]
            self.buffer = [row for row in X[idx[:n_fit]]]
        X_fit = np.vstack(self.buffer[-600:])
        self.forest = IsolationForest(n_estimators=200, random_state=self.np_seed).fit(X_fit)
        self.buffer_scores = -self.forest.score_samples(self.calib_X)
        self.forest_round = self.round_no

    def score(self, z_vec: np.ndarray) -> float:
        return float(-self.forest.score_samples(z_vec.reshape(1, -1))[0])

    def thresholds(self, arm_idx: int, exposure: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
        if self.forest is None or len(self.buffer_scores) == 0:
            return None, None
        t_alert = float(np.quantile(self.buffer_scores, self.bandit.ARMS[arm_idx]))
        q_watch = Q_WATCH - Q_WATCH_EXPOSURE_DROP * min(1.0, exposure)
        t_watch = float(np.quantile(self.buffer_scores, q_watch))
        return t_watch, max(t_watch, t_alert)

    def _fit_beta(self):
        """Ambient slope of the thermal reading, estimated from healthy (Tier 0) readings only,
        demeaned per animal and pooled."""
        Ts, rs = [], []
        for obs in self._beta_obs.values():
            if len(obs) < 5:
                continue
            t = np.array([o[0] for o in obs])
            v = np.array([o[1] for o in obs])
            Ts.extend(t - t.mean())
            rs.extend(v - v.mean())
        if len(Ts) >= 40 and np.var(Ts) > 1e-6:
            self.beta_thermal = float(np.polyfit(np.array(Ts), np.array(rs), 1)[0])

    def _context_vector(self, ambient: float) -> np.ndarray:
        recent = [a for a in self.alerts if a.source == "raceway" and a.round_no > self.round_no - 7]
        rate = min(1.0, len(recent) / max(1, len(self.animals)))
        return np.array([1.0, (ambient - T_REF) / 10.0, rate])

    # ----- simulation of sensing ------------------------------------------

    def _sense(self, animal: Animal, ambient: float, spike: bool) -> Dict[str, float]:
        eff = _event_effect(animal.event)
        raw = {}
        for f in FEATURES:
            v = animal.base[f] + eff[f] + self.rng.gauss(0, NOISE_SD[f])
            if f == "thermal":
                v += BETA_THERMAL_TRUE * (ambient - T_REF)
            if spike and f in ("thermal", "cough"):
                v += 0.8 if f == "thermal" else 1.5
            raw[f] = v
        raw["cough"] = max(0.0, raw["cough"])
        raw["gait"] = min(100.0, max(0.0, raw["gait"]))
        raw["lesion"] = min(100.0, max(0.0, raw["lesion"]))
        raw["speed"] = max(0.2, raw["speed"])
        return raw

    def _residual(self, raw: Dict[str, float], ambient: float) -> Dict[str, float]:
        resid = dict(raw)
        resid["thermal"] = raw["thermal"] - self.beta_thermal * (ambient - T_REF)
        return resid

    def _zscores(self, animal: Animal, resid: Dict[str, float]) -> Optional[Dict[str, float]]:
        if any(len(animal.window[f]) < MIN_OBS for f in FEATURES):
            return None
        z = {}
        for f in FEATURES:
            w = np.array(animal.window[f])
            sd = max(float(w.std(ddof=1)), MIN_SD[f])
            z[f] = float((resid[f] - w.mean()) / sd)
        return z

    @staticmethod
    def pattern_scores(z: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for name, w in PATTERNS.items():
            out[name] = round(sum(max(0.0, z[f]) * wt for f, wt in w.items()), 2)
        return out

    def dominant_pattern(self, z: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        ps = self.pattern_scores(z)
        name = max(ps, key=ps.get)
        return (name if ps[name] >= 1.0 else "Unspecified deviation"), ps

    # ----- events ----------------------------------------------------------

    def _progress_events(self, warmup: bool):
        for a in self.animals.values():
            ev = a.event
            if ev is not None:
                ev["day"] += 1
                td = ev.get("treated_day")
                if td is not None and ev["day"] - td >= RECOVERY_CLEARANCE_DAYS:
                    a.event = None
                    a.recoveries += 1
                    if a.isolated:
                        self._release(a, reason="recovered")
                elif td is None and ev["day"] > ev["duration"]:
                    a.event = None
        if warmup:
            return
        pen_size: Dict[str, int] = {}
        for b in self.animals.values():
            pen_size[b.pen] = pen_size.get(b.pen, 0) + 1
        for a in self.animals.values():
            if a.status != "active" or a.event is not None:
                continue
            base_p = self.daily_events / max(1, len(self.animals))
            contagion = 0.0
            # contagion within a pen from an un-isolated febrile case (scaled by pen size)
            for b in self.animals.values():
                if (b is not a and b.pen == a.pen and not b.isolated and b.event
                        and b.event["pattern"] == "Febrile infectious" and b.event["kind"] == "disease"):
                    contagion += 0.15 / pen_size[a.pen]
            p_start = base_p + contagion
            if self.rng.random() < p_start:
                spreading = contagion > 0
                if spreading or self.rng.random() < 0.7:
                    kind = "disease"
                else:
                    kind = "transient"
                if spreading:
                    pattern = "Febrile infectious"
                else:
                    pattern = self.rng.choices(list(PATTERNS), weights=[0.25, 0.25, 0.30, 0.20])[0]
                a.event = {"pattern": pattern, "kind": kind, "day": 0,
                           "duration": self.rng.randint(6, 12) if kind == "disease" else self.rng.randint(2, 3),
                           "treated_day": None, "start_round": self.round_no}

    def _release(self, animal: Animal, reason: str):
        animal.isolated = False
        animal.isolation_start = None
        animal.recent_exceed.clear()

    # ----- exposure --------------------------------------------------------

    def exposure(self, aid: str, tier2_now: set, pass_times: Dict[str, float]) -> Tuple[float, List[Tuple[str, float]]]:
        passage = 0.0
        pen = 0.0
        contacts: List[Tuple[str, float]] = []
        me = self.animals[aid]
        for j in self.animals.values():
            if j.id == aid:
                continue
            e_pass = 0.0
            if j.id in tier2_now and aid in pass_times and j.id in pass_times:
                e_pass = math.exp(-abs(pass_times[aid] - pass_times[j.id]) / TAU_SECONDS)
                passage += e_pass
            e_pen = 0.0
            if j.last_flag_round is not None and j.pen == me.pen and not self._cleared(j.id):
                age = self.round_no - j.last_flag_round
                if 0 <= age <= 7:
                    e_pen = math.exp(-age / TAU_DAYS)
                    pen += e_pen
            if e_pass + e_pen > 0.05:
                contacts.append((j.id, round(0.5 * e_pass + 0.5 * e_pen, 2)))
        total = min(1.0, 0.5 * min(1.0, passage) + 0.5 * min(1.0, pen))
        return round(total, 3), sorted(contacts, key=lambda t: -t[1])

    # ----- one handling round ---------------------------------------------

    def _cleared(self, aid: str) -> bool:
        """True if the most recent alert for this animal was resolved as 'no problem found' / fault."""
        recs = [r for r in self.alerts if r.animal_id == aid]
        return bool(recs) and recs[-1].resolved and recs[-1].outcome in (OUT_NEG, OUT_FAULT)

    def step(self, ambient: float, warmup: bool = False) -> dict:
        self.round_no += 1
        r = self.round_no
        self.last_ambient = ambient
        self._progress_events(warmup)
        if not warmup:
            for rec in self.alerts:
                if not rec.resolved and self.animals[rec.animal_id].isolated:
                    rec.isolation_rounds += 1

        ctx_x = self._context_vector(ambient)
        if warmup:
            arm = self.bandit.BASELINE
            sel = {"arm": arm, "quantile": self.bandit.ARMS[arm], "allowed": [self.bandit.ARMS[arm]], "slack": 0.0}
        else:
            arm, sel = self.bandit.select(ctx_x)
        self.last_arm, self.last_context = arm, ctx_x

        handling = warmup or (r % self.context.handling_interval == 0)
        observed: Dict[str, dict] = {}
        round_warnings: List[dict] = []
        pass_times: Dict[str, float] = {}

        if handling:
            order = [a for a in self.animals.values() if a.status == "active" and not a.isolated]
            self.rng.shuffle(order)
            t = 0.0
            for animal in order:
                t += self.rng.uniform(4, 10)
                pass_times[animal.id] = t
                if warmup:
                    raw = self._sense(animal, ambient, False)
                    self._warm_raw[animal.id].append((ambient, raw))
                    animal.last_valid_round = r
                    continue
                if self.rng.random() < P_DROPOUT:
                    self.counters["dropouts"] += 1
                    round_warnings.append({"round": r, "animal": animal.id, "type": "sensor dropout",
                                           "detail": "no valid reading this pass; not treated as healthy"})
                    continue
                id_conf = self.rng.uniform(0.92, 1.0)
                if self.rng.random() < P_ID_UNCERTAIN:
                    id_conf = self.rng.uniform(0.5, 0.78)
                if id_conf < ID_CONF_MIN:
                    self.counters["id_uncertain"] += 1
                    round_warnings.append({"round": r, "animal": animal.id, "type": "identity uncertain",
                                           "detail": f"face-ID confidence {id_conf:.2f}: reading not linked to history, "
                                                     f"not used for learning; check identity"})
                    continue
                spike = self.rng.random() < P_TRANSIENT_SPIKE
                if spike:
                    self.counters["transients"] += 1
                raw = self._sense(animal, ambient, spike)
                resid = self._residual(raw, ambient)
                z = self._zscores(animal, resid)
                observed[animal.id] = {"raw": raw, "resid": resid, "z": z, "id_conf": id_conf}
                animal.last_valid_round = r

        if warmup:
            return {"round": r, "handling": True}

        # --- score every valid observation ---
        tier2_now: set = set()
        scored: Dict[str, dict] = {}
        batch_ids = [aid for aid, obs in observed.items() if obs["z"] is not None and self.forest is not None]
        S_map: Dict[str, float] = {}
        if batch_ids:
            Zm = np.array([[observed[a]["z"][f] for f in FEATURES] for a in batch_ids])
            S_map = dict(zip(batch_ids, (-self.forest.score_samples(Zm)).tolist()))
        for aid, obs in observed.items():
            animal = self.animals[aid]
            if obs["z"] is None or self.forest is None:
                scored[aid] = {"tier": "building baseline", "S": None, "pattern": None}
                for f in FEATURES:
                    animal.window[f].append(obs["resid"][f])
                continue
            z_vec = np.array([obs["z"][f] for f in FEATURES])
            S = S_map[aid]
            _, theta_alert = self.thresholds(arm)
            exceed = S >= theta_alert
            animal.recent_exceed.append(bool(exceed))
            persistent = exceed and sum(animal.recent_exceed) >= K_PERSIST
            pattern, pscores = self.dominant_pattern(obs["z"])
            scored[aid] = {"S": S, "exceed": bool(exceed), "persistent": bool(persistent), "pattern": pattern,
                           "pscores": pscores, "z_vec": z_vec}
            if persistent:
                tier2_now.add(aid)

        # --- tiers, exposure, alerts ---
        for aid, sc in scored.items():
            if sc["S"] is None:
                continue
            obs = observed[aid]
            animal = self.animals[aid]
            E, contacts = self.exposure(aid, tier2_now, pass_times)
            t_watch, t_alert = self.thresholds(arm, E)
            if sc["persistent"]:
                tier = 2
            elif sc["exceed"] or sc["S"] >= t_watch:
                tier = 1
            else:
                tier = 0
            sc.update({"tier": tier, "E": E, "contacts": contacts, "theta_watch": t_watch, "theta_alert": t_alert})
            open_alert = any(rec.animal_id == aid and not rec.resolved for rec in self.alerts)

            if tier == 0:  # only healthy readings enter the baseline and the healthy buffer
                for f in FEATURES:
                    animal.window[f].append(obs["resid"][f])
                self.buffer.append(sc["z_vec"])
                self._beta_obs.setdefault(aid, []).append((ambient, obs["raw"]["thermal"]))
                self._beta_obs[aid] = self._beta_obs[aid][-60:]

            if tier == 2 and not open_alert:
                auto = (self.context.gate_mode == GATE_AUTO) and not self.context.gate_paused
                rec = AlertRecord(
                    id=len(self.alerts) + 1, animal_id=aid, round_no=r, source="raceway",
                    pattern=sc["pattern"], S=round(sc["S"], 3), theta_alert=round(t_alert, 3),
                    z={f: round(obs["z"][f], 2) for f in FEATURES},
                    raw={f: round(obs["raw"][f], 3) for f in FEATURES},
                    resid=dict(obs["resid"]), arm_idx=arm, context=[float(v) for v in ctx_x],
                    gate_action="diverted" if auto else "inspection requested",
                    exposure_contacts=contacts,
                    truth_disease=bool(animal.event and animal.event["kind"] == "disease"),
                    truth_start_round=animal.event["start_round"] if animal.event else None,
                    truth_pattern=animal.event["pattern"] if animal.event else None,
                    pass_s=pass_times.get(aid, 0.0),
                )
                if rec.is_reportable:
                    rec.suspicion = True
                    rec.suspicion_round = r
                elif self.context.suspicion_rule == SUSPICION_RULES[1]:
                    rec.suspicion = True
                    rec.suspicion_round = r
                self.alerts.append(rec)
                animal.last_flag_round = r
                if auto:
                    animal.isolated = True
                    animal.isolation_start = r
            elif tier == 2:
                animal.last_flag_round = r

        # --- passage feed and per-animal reading history (for the UI) ---
        for aid, obs in observed.items():
            sc = scored.get(aid, {})
            row = {"round": r, "pass_s": pass_times.get(aid, 0.0), "animal": aid, "id_conf": obs["id_conf"],
                   "tier": sc.get("tier"), "S": sc.get("S")}
            self.feed.append(row)
            rd = {"round": r, "pass_s": pass_times.get(aid, 0.0), "raw": {f: obs["raw"][f] for f in FEATURES},
                  "z": obs["z"], "S": sc.get("S"), "tier": sc.get("tier"), "ambient": ambient}
            self.readings[aid].append(rd)
            self.readings[aid] = self.readings[aid][-40:]
        self.feed = self.feed[-1500:]

        # --- model upkeep: periodic refit on healthy readings only ---
        if r % REFIT_EVERY == 0:
            self._fit_forest(force=True)

        # --- animals not observed ---
        limit = 2 * self.context.handling_interval + 2
        for a in self.animals.values():
            if a.status == "active" and not a.isolated and (r - a.last_valid_round) > limit:
                round_warnings.append({"round": r, "animal": a.id, "type": "not observed",
                                       "detail": f"no valid reading for {r - a.last_valid_round} rounds: needs a "
                                                 f"manual check (missing data is not evidence of health)"})

        self.warnings.extend(round_warnings)
        for aid, sc in scored.items():
            if sc.get("S") is not None:
                self.history[aid].append({"round": r, "S": sc["S"], "tier": sc["tier"],
                                          "theta_alert": sc["theta_alert"], "theta_watch": sc["theta_watch"]})
        self.round_log.append({"round": r, "handling": handling, "arm_quantile": sel["quantile"],
                               "allowed_arms": sel["allowed"], "n_tier2": len(tier2_now),
                               "n_warnings": len(round_warnings)})
        self.last_tiers = scored
        return {"round": r, "handling": handling, "ambient": ambient, "arm": sel,
                "scored": scored, "observed": observed, "warnings": round_warnings}

    # ----- human actions ---------------------------------------------------

    def get_alert(self, alert_id: int) -> AlertRecord:
        return next(a for a in self.alerts if a.id == alert_id)

    def pending(self) -> List[AlertRecord]:
        return [a for a in self.alerts if not a.resolved]

    def notify(self, alert_id: int, vet: bool = False, authority: bool = False):
        """Record that a HUMAN contacted the vet / notified the authority. The system never does this itself."""
        rec = self.get_alert(alert_id)
        if vet and rec.vet_contacted_round is None:
            rec.vet_contacted_round = self.round_no
        if authority and rec.authority_notified_round is None:
            rec.authority_notified_round = self.round_no
            if rec.vet_contacted_round is None:
                rec.vet_contacted_round = self.round_no

    def record_pen_test(self, alert_id: int, result: str):
        if result not in ("positive", "negative"):
            raise ValueError("result must be 'positive' or 'negative'")
        self.get_alert(alert_id).pen_test = result

    def record_outcome(self, alert_id: int, outcome: str, evidence: str, notes: str = ""):
        rec = self.get_alert(alert_id)
        if outcome not in OUTCOMES:
            raise ValueError("unknown outcome")
        animal = self.animals[rec.animal_id]
        if rec.is_reportable and outcome == OUT_NEG and evidence in (EV_FARMER, EV_PENTEST):
            raise SequenceViolation(
                "A reportable-pattern suspicion cannot be cleared by a farmer observation or a presumptive "
                "pen-side test. A negative on-farm result lowers, but does not remove, the suspicion: "
                "veterinary assessment is required.")
        if rec.is_reportable and outcome == OUT_POS:
            if evidence not in (EV_VET, EV_LAB):
                raise SequenceViolation(
                    "Establishing a reportable case needs veterinary assessment or reference-laboratory evidence. "
                    "A farmer observation or rapid test only strengthens the suspicion.")
            if rec.authority_notified_round is None:
                raise SequenceViolation(
                    "Notification of the competent authority is owed on suspicion and must already be recorded "
                    "(Regulation (EU) 2016/429). It must not wait for test results.")
        if rec.is_reportable and outcome == OUT_FAULT and evidence not in (EV_VET, EV_LAB):
            raise SequenceViolation(
                "A reportable-pattern suspicion can only be dismissed as a sensor/identification fault "
                "after veterinary assessment.")
        rec.notes = notes
        rec.evidence = evidence
        rec.outcome = outcome
        if outcome == OUT_UNC:
            rec.resolved = False
            return rec
        rec.resolved = True
        rec.resolved_round = self.round_no
        w = SOURCE_WEIGHT.get(evidence, 0.4)

        if outcome == OUT_POS:
            self.bandit.update(rec.arm_idx, np.array(rec.context), REWARDS["TP"], w)
            if rec.is_reportable:
                animal.status = "official"      # handed to the official pathway; outside this system's scope
                animal.isolated = True
            elif animal.event is not None:
                animal.event["treated_day"] = animal.event["day"]
        elif outcome == OUT_NEG:
            self.bandit.update(rec.arm_idx, np.array(rec.context), REWARDS["FP"], w)
            if evidence in (EV_VET, EV_LAB):
                for f in FEATURES:             # natural variation: allow it into the baseline
                    animal.window[f].append(rec.resid[f])
            animal.isolated = False
            animal.isolation_start = None
            animal.recent_exceed.clear()
        elif outcome == OUT_FAULT:
            animal.isolated = False
            animal.isolation_start = None
            animal.recent_exceed.clear()
        return rec

    def auto_resolve_pending(self) -> List[str]:
        """Demo helper: simulate vets assessing every pending alert using the simulated ground truth."""
        log = []
        for rec in list(self.pending()):
            animal = self.animals[rec.animal_id]
            disease = bool(rec.truth_disease)
            if rec.is_reportable:
                self.notify(rec.id, vet=True, authority=True)
            else:
                self.notify(rec.id, vet=True)
            if disease:
                self.record_outcome(rec.id, OUT_POS, EV_VET, "simulated vet assessment")
                log.append(f"{rec.animal_id}: health problem established ({rec.pattern})")
            else:
                self.record_outcome(rec.id, OUT_NEG, EV_VET, "simulated vet assessment")
                log.append(f"{rec.animal_id}: no problem found")
        return log

    # ----- audits ----------------------------------------------------------

    def run_audit(self, fraction: Optional[float] = None) -> List[dict]:
        """Clinical check of a random sample of animals that are NOT currently flagged."""
        frac = self.context.audit_fraction if fraction is None else fraction
        pool = [a for a in self.animals.values()
                if a.status == "active" and not a.isolated
                and not any(rec.animal_id == a.id and not rec.resolved for rec in self.alerts)]
        if not pool:
            return []
        n = max(1, round(frac * len(pool)))
        sample = self.rng.sample(pool, min(n, len(pool)))
        results = []
        for a in sample:
            self.audit_n += 1
            sick = bool(a.event and a.event["kind"] == "disease" and a.event["day"] >= 2 and a.event.get("treated_day") is None)
            x = self.last_context
            if sick:
                self.audit_pos += 1
                self.bandit.update(self.last_arm, x, REWARDS["FN"], SOURCE_WEIGHT[EV_AUDIT])
                pat = a.event["pattern"]
                rec = AlertRecord(
                    id=len(self.alerts) + 1, animal_id=a.id, round_no=self.round_no, source="audit",
                    pattern=pat, S=float("nan"), theta_alert=float("nan"), z={}, raw={}, resid={},
                    arm_idx=self.last_arm, context=[float(v) for v in x], gate_action="found by audit",
                    suspicion=pat in REPORTABLE_PATTERNS,
                    suspicion_round=self.round_no if pat in REPORTABLE_PATTERNS else None,
                    truth_disease=True, truth_start_round=a.event["start_round"],
                    truth_pattern=pat,
                )
                self.alerts.append(rec)
                a.isolated = True
                a.isolation_start = self.round_no
                a.last_flag_round = self.round_no
                results.append({"animal": a.id, "result": "MISSED CASE found", "pattern": pat})
            else:
                self.bandit.update(self.last_arm, x, REWARDS["TN"], SOURCE_WEIGHT[EV_AUDIT])
                results.append({"animal": a.id, "result": "healthy", "pattern": None})
        self.audit_log.extend({"round": self.round_no, **r} for r in results)
        return results

    def audit_bound(self) -> dict:
        n, pos = self.audit_n, self.audit_pos
        if n == 0:
            return {"n": 0, "pos": 0, "text": "no audits yet", "bound": None}
        if pos == 0:
            b = min(1.0, 3.0 / n)
            return {"n": n, "pos": 0, "bound": b,
                    "text": f"0 misses in {n} audited animals: 95% upper bound on the miss rate among "
                            f"UNFLAGGED animals is about {b:.0%} (rule of three; assumes independent samples)."}
        return {"n": n, "pos": pos, "bound": pos / n,
                "text": f"{pos} miss(es) in {n} audited animals (observed miss rate {pos / n:.0%} among unflagged animals)."}

    # ----- evaluation ------------------------------------------------------

    def metrics(self) -> dict:
        raceway = [a for a in self.alerts if a.source == "raceway"]
        days = max(1, self.round_no)
        resolved = [a for a in raceway if a.resolved and a.outcome in (OUT_POS, OUT_NEG)]
        tp = [a for a in resolved if a.outcome == OUT_POS]
        fp = [a for a in resolved if a.outcome == OUT_NEG]
        lat = [a.round_no - a.truth_start_round for a in tp if a.truth_start_round is not None]
        unnecessary = [a for a in fp if a.gate_action == "diverted"]
        susp = [a for a in self.alerts if a.suspicion]
        notif_lags = [a.authority_notified_round - a.suspicion_round for a in susp
                      if a.authority_notified_round is not None and a.suspicion_round is not None]
        unnotified = [a for a in susp if a.authority_notified_round is None]
        tp_known = [a for a in tp if a.truth_pattern is not None]
        pattern_hits = [a for a in tp_known if a.pattern == a.truth_pattern]
        return {
            "pattern_match_rate": round(len(pattern_hits) / len(tp_known), 2) if tp_known else None,
            "pattern_match_n": len(tp_known),
            "false_alarm_rate": round(len(fp) / len(resolved), 2) if resolved else None,
            "rounds": self.round_no,
            "raceway_alerts": len(raceway),
            "alerts_per_round": round(len(raceway) / days, 2),
            "resolved_alerts": len(resolved),
            "precision_among_resolved": round(len(tp) / len(resolved), 2) if resolved else None,
            "mean_detection_latency_rounds": round(sum(lat) / len(lat), 1) if lat else None,
            "unnecessary_separations": len(unnecessary),
            "unnecessary_isolation_rounds": sum(a.isolation_rounds for a in unnecessary),
            "unresolved_alerts": len([a for a in raceway if not a.resolved]),
            "suspicions": len(susp),
            "suspicions_not_yet_notified": len(unnotified),
            "mean_rounds_suspicion_to_notification": round(sum(notif_lags) / len(notif_lags), 1) if notif_lags else None,
            "audit": self.audit_bound(),
            "data_quality": dict(self.counters),
            "audit_missed_cases": self.audit_pos,
        }

    # ----- outputs for humans ---------------------------------------------

    def explain(self, rec: AlertRecord) -> str:
        if rec.source == "audit":
            return "Found by a random clinical audit of an unflagged animal (a case the station had not flagged)."
        parts = []
        for f in sorted(rec.z, key=lambda k: -abs(rec.z[k])):
            if abs(rec.z[f]) >= 2.0:
                parts.append(f"{FEATURE_LABELS[f].lower()} {rec.raw[f]:.2f}{FEATURE_UNITS[f]} "
                             f"(z = {rec.z[f]:+.1f} vs this animal's own baseline)")
        body = "; ".join(parts) if parts else "a combination of smaller deviations across several features"
        return (f"Health alert (not a diagnosis). Anomaly score S = {rec.S:.2f} against alert threshold "
                f"{rec.theta_alert:.2f}, above threshold on {K_PERSIST} of the last {N_PERSIST} passes. "
                f"Strongest signals: {body}. Closest pattern: {rec.pattern}.")

    def explain_watch(self, aid: str) -> str:
        rows = self.readings.get(aid) or []
        if not rows or rows[-1]["z"] is None:
            return "No scored reading yet."
        z, raw = rows[-1]["z"], rows[-1]["raw"]
        parts = [f"{FEATURE_LABELS[f].lower()} {raw[f]:.2f}{FEATURE_UNITS[f]} (z = {z[f]:+.1f})"
                 for f in sorted(z, key=lambda k: -abs(z[k])) if abs(z[f]) >= 1.5]
        return "; ".join(parts) if parts else "small deviations across several features"

    def build_dossier(self, alert_id: int) -> dict:
        rec = self.get_alert(alert_id)
        hist = [h for h in self.history.get(rec.animal_id, [])][-10:]
        return {
            "NOTE": "Decision-support dossier. Not a diagnosis. Anomaly scores are statistical deviations from "
                    "this animal's own baseline. Official confirmation remains with the reference laboratory.",
            "animal_id": rec.animal_id,
            "alert_round": rec.round_no,
            "alert_source": rec.source,
            "closest_pattern": rec.pattern,
            "reportable_pattern": rec.is_reportable,
            "legal_suspicion_flag": rec.suspicion,
            "suspicion_round": rec.suspicion_round,
            "vet_contacted_round": rec.vet_contacted_round,
            "authority_notified_round": rec.authority_notified_round,
            "anomaly_score": None if rec.source == "audit" else rec.S,
            "alert_threshold": None if rec.source == "audit" else rec.theta_alert,
            "feature_z_scores": rec.z,
            "raw_readings": rec.raw,
            "score_history_last_10_passes": [{"round": h["round"], "S": round(h["S"], 3), "tier": h["tier"]} for h in hist],
            "contacts_with_flagged_animals": [{"animal": a, "exposure": e} for a, e in rec.exposure_contacts],
            "pen_side_test_presumptive": rec.pen_test,
            "evidence_recorded": rec.evidence,
            "outcome_recorded": rec.outcome,
            "notes": rec.notes,
        }


# ---------------------------------------------------------------------------
# Cost model (ALL PLACEHOLDERS -- replace with supplier quotes before use)
# ---------------------------------------------------------------------------

DEFAULT_STATION_COMPONENTS = {
    "Thermal cameras (2, both sides of the race)": 0,
    "RGB-depth camera + face-ID processing": 0,
    "Directional microphone array": 0,
    "Electric gate and actuator": 0,
    "Edge computer + local encrypted storage": 0,
    "Power (battery/solar), cabling, weather sealing": 0,
    "Mobile sorting yard kit (extensive farms only)": 0,
}


def estimate_cost(components: Dict[str, float], n_animals: int, system_type: str,
                  sentinel_pct: float, cost_per_collar: float, enrolment_per_animal: float) -> dict:
    station = sum(v for k, v in components.items()
                  if not (k.startswith("Mobile sorting") and system_type == SYSTEM_INTENSIVE))
    sentinel = math.ceil(n_animals * sentinel_pct / 100.0) * cost_per_collar
    enrol = n_animals * enrolment_per_animal
    wearables_all = n_animals * cost_per_collar
    return {
        "station": station, "sentinel_collars_temporary": sentinel, "face_enrolment": enrol,
        "total_this_design": station + sentinel + enrol,
        "alternative_collars_on_all": wearables_all,
    }


# ---------------------------------------------------------------------------
# Privacy: secondary-use / regional summaries (statutory notification is NOT anonymised)
# ---------------------------------------------------------------------------

def anonymize_id(animal_id: str, salt: str = "demo-salt") -> str:
    return "HASH-" + hashlib.sha256((salt + animal_id).encode()).hexdigest()[:8].upper()


def coarse_cell(lat: float, lon: float, grid_km: float = 25.0) -> Tuple[float, float]:
    cell = grid_km / 111.0
    return round(round(lat / cell) * cell, 2), round(round(lon / cell) * cell, 2)


def regional_summary(alerts: List[AlertRecord], farm_lat: float = 39.15, farm_lon: float = 26.30) -> List[dict]:
    """What may leave the farm for secondary use: pattern counts per coarse grid cell, no IDs, no exact location.
    A statutory notification of a suspected Category A disease is a different channel and identifies the holding."""
    cell = coarse_cell(farm_lat, farm_lon)
    counts: Dict[str, int] = {}
    for a in alerts:
        counts[a.pattern] = counts.get(a.pattern, 0) + 1
    return [{"grid_cell": cell, "pattern": p, "alerts": n} for p, n in sorted(counts.items())]
