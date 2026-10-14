"""FitMind backend: FastAPI + SQLAlchemy (SQLite locally, PostgreSQL in production)."""
import os, time, json, hashlib, secrets
from typing import List, Optional
import jwt
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DB = os.getenv("DATABASE_URL", "sqlite:///./fitmind.db")
if DB.startswith("postgres://"):
    DB = DB.replace("postgres://", "postgresql://", 1)
engine = create_engine(DB, connect_args={"check_same_thread": False} if DB.startswith("sqlite") else {})
Session = sessionmaker(bind=engine)
Base = declarative_base()
SECRET = os.getenv("SECRET_KEY", "change-me-in-production")

# ---------- Database tables ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    salt = Column(String)
    pw_hash = Column(String)

class Profile(Base):
    __tablename__ = "profiles"
    user_id = Column(Integer, primary_key=True)
    data = Column(Text)          # JSON: profile form + senior flag

class WeightLog(Base):
    __tablename__ = "weight_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    ts = Column(Float)           # milliseconds
    weight = Column(Float)

Base.metadata.create_all(engine)

# ---------- Helpers ----------
def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()

def make_token(uid):
    return jwt.encode({"sub": str(uid), "exp": time.time() + 7 * 86400}, SECRET, algorithm="HS256")

def db():
    s = Session()
    try:
        yield s
    finally:
        s.close()

def current_user(authorization: str = Header(default=""), s=Depends(db)):
    try:
        uid = int(jwt.decode(authorization.replace("Bearer ", ""), SECRET, algorithms=["HS256"])["sub"])
    except Exception:
        raise HTTPException(401, "Please log in again")
    return uid

class AuthIn(BaseModel):
    email: str
    password: str
    name: str = ""

class StateIn(BaseModel):
    p: Optional[dict] = None
    logs: List[dict] = []
    senior: bool = False

# ---------- API routes ----------
app = FastAPI(title="FitMind API", description="AI fitness platform. Interactive docs at /docs")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/register")
def register(b: AuthIn, s=Depends(db)):
    email = b.email.strip().lower()
    if "@" not in email or len(b.password) < 6:
        raise HTTPException(400, "Enter a valid email and a password of 6+ characters")
    if s.query(User).filter_by(email=email).first():
        raise HTTPException(400, "Email already registered. Please log in")
    salt = secrets.token_hex(8)
    u = User(email=email, name=b.name, salt=salt, pw_hash=hash_pw(b.password, salt))
    s.add(u); s.commit()
    return {"token": make_token(u.id)}

@app.post("/api/login")
def login(b: AuthIn, s=Depends(db)):
    u = s.query(User).filter_by(email=b.email.strip().lower()).first()
    if not u or u.pw_hash != hash_pw(b.password, u.salt):
        raise HTTPException(400, "Wrong email or password")
    return {"token": make_token(u.id)}

@app.get("/api/state")
def get_state(uid=Depends(current_user), s=Depends(db)):
    pr = s.get(Profile, uid)
    blob = json.loads(pr.data) if pr else {}
    logs = s.query(WeightLog).filter_by(user_id=uid).order_by(WeightLog.ts).all()
    return {"p": blob.get("p"), "senior": blob.get("senior", False),
            "logs": [{"d": l.ts, "w": l.weight} for l in logs]}

@app.put("/api/state")
def put_state(b: StateIn, uid=Depends(current_user), s=Depends(db)):
    data = json.dumps({"p": b.p, "senior": b.senior})
    pr = s.get(Profile, uid)
    if pr: pr.data = data
    else: s.add(Profile(user_id=uid, data=data))
    s.query(WeightLog).filter_by(user_id=uid).delete()
    for l in b.logs:
        s.add(WeightLog(user_id=uid, ts=float(l["d"]), weight=float(l["w"])))
    s.commit()
    return {"saved": True}

@app.post("/api/ai/recommend")
def ai_recommend(b: StateIn, uid=Depends(current_user)):
    import ml
    try:
        return ml.recommend(b.p)
    except Exception:
        raise HTTPException(400, "Save your profile first")

@app.get("/api/analytics")
def analytics(uid=Depends(current_user), s=Depends(db)):
    import numpy as np
    logs = s.query(WeightLog).filter_by(user_id=uid).order_by(WeightLog.ts).all()
    if not logs:
        raise HTTPException(404, "No data")
    w = np.array([l.weight for l in logs]); t = np.array([l.ts for l in logs]) / 864e5
    weekly = float(np.polyfit(t, w, 1)[0] * 7) if len(w) >= 3 and t.max() > t.min() else 0.0
    pr = s.get(Profile, uid); goal = (json.loads(pr.data).get("p") or {}).get("goal", "keep") if pr else "keep"
    ok = (goal == "lose" and weekly < -0.1) or (goal == "gain" and weekly > 0.1) or (goal == "keep" and abs(weekly) < 0.3)
    status = "On track with your goal. Keep going." if ok else "Not yet on track. Check the adaptive calorie advice."
    if len(w) < 3: status = "Log at least 3 weights to see your trend."
    return {"current": round(float(w[-1]), 1), "change": round(float(w[-1] - w[0]), 1), "weekly": round(weekly, 2), "entries": len(w), "status": status}

@app.get("/api/ai/insights")
def insights(uid=Depends(current_user), s=Depends(db)):
    import ml
    pr = s.get(Profile, uid)
    p = json.loads(pr.data).get("p") if pr else None
    if not p:
        raise HTTPException(400, "Save your profile first")
    w = [l.weight for l in s.query(WeightLog).filter_by(user_id=uid).order_by(WeightLog.ts)]
    return {**ml.risk(p), "anomalies": ml.anomalies(w)}

# Frontend (served by the same server)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
