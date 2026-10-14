"""Server-side ML: preprocessing (StandardScaler) + neural network (MLPRegressor)."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

NAMES = ["age", "height_cm", "weight_kg", "activity", "goal_lose", "goal_gain", "male", "has_condition"]
ACT = {"lose": -450, "gain": 300, "keep": 0}

def features(p):
    """Feature extraction from the profile dictionary."""
    g = p.get("goal", "keep")
    return [float(p["age"]), float(p["h"]), float(p["w"]), float(p["act"]),
            1.0 if g == "lose" else 0.0, 1.0 if g == "gain" else 0.0,
            1.0 if p.get("g") == "Male" else 0.0, 0.0 if p.get("cond", "None") == "None" else 1.0]

def bmr_tdee(p):
    bmr = 10 * p["w"] + 6.25 * p["h"] - 5 * p["age"] + (5 if p.get("g") == "Male" else -161)
    return bmr, bmr * p["act"]

def _train():
    rng = np.random.default_rng(0)
    X, Y = [], []
    for _ in range(2000):
        p = dict(age=rng.uniform(18, 80), h=rng.uniform(150, 195), w=rng.uniform(45, 110), act=rng.uniform(1.2, 1.725),
                 goal=str(rng.choice(["lose", "gain", "keep"])), g=str(rng.choice(["Male", "Female"])),
                 cond="Heart" if rng.random() < .3 else "None")
        kcal = bmr_tdee(p)[1] + ACT[p["goal"]]
        inten = min(1, max(.1, (p["act"] - 1.2) * 1.5 + (.2 if p["goal"] == "lose" else 0)
                           - (.3 if p["age"] > 60 else 0) - (.25 if p["cond"] != "None" else 0) + .2))
        X.append(features(p)); Y.append([kcal / 3500, inten])
    m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0))
    m.fit(np.array(X), np.array(Y))
    return m

MODEL = _train()

def recommend(p):
    x = np.array([features(p)])
    scaled = MODEL[0].transform(x)[0]
    out = MODEL.predict(x)[0]
    bmr, tdee = bmr_tdee(p)
    return {"kcal": int(round(out[0] * 3500)), "intensity": float(min(1, max(0, out[1]))),
            "bmi": round(p["w"] / (p["h"] / 100) ** 2, 1), "bmr": int(bmr), "tdee": int(tdee),
            "features": [{"name": n, "raw": round(r, 2), "scaled": round(float(s), 2)} for n, r, s in zip(NAMES, x[0], scaled)]}


# ---------- Advanced ML: explainable risk classifier + anomaly detection ----------
from sklearn.ensemble import RandomForestClassifier, IsolationForest
RN = NAMES + ["bmi"]
LABELS = ["Low risk", "Moderate risk", "High risk"]

def _f2(p):
    return features(p) + [p["w"] / (p["h"] / 100) ** 2]

def _train_rf():
    rng = np.random.default_rng(1)
    X, y = [], []
    for _ in range(2500):
        p = dict(age=rng.uniform(18, 80), h=rng.uniform(150, 195), w=rng.uniform(45, 120), act=rng.uniform(1.2, 1.725),
                 goal="keep", g="Male", cond="Heart" if rng.random() < .3 else "None")
        f = _f2(p); bmi = f[-1]
        score = (bmi >= 30) * 2 + (bmi >= 25) + (p["age"] > 60) + (1.5 if p["cond"] != "None" else 0) + (p["act"] < 1.3)
        X.append(f); y.append(0 if score <= 1 else 1 if score <= 3 else 2)
    return RandomForestClassifier(150, random_state=0).fit(X, y)

RF = _train_rf()

def risk(p):
    f = np.array([_f2(p)])
    probs = {LABELS[int(c)]: round(float(v), 2) for c, v in zip(RF.classes_, RF.predict_proba(f)[0])}
    imp = sorted(zip(RN, RF.feature_importances_), key=lambda t: -t[1])[:5]
    return {"risk": LABELS[int(RF.predict(f)[0])], "probs": probs,
            "importance": [{"name": n, "value": round(float(v), 3)} for n, v in imp]}

def anomalies(w):
    """Isolation Forest flags unusual weight changes between entries."""
    if len(w) < 6:
        return []
    d = np.diff(np.array(w)).reshape(-1, 1)
    flag = IsolationForest(contamination=.15, random_state=0).fit_predict(d)
    return [{"entry": i + 2, "change": round(float(d[i][0]), 1)} for i in range(len(d)) if flag[i] == -1]
