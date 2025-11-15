from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import os
from decimal import Decimal, InvalidOperation
import sys
import uuid
import asyncio
import importlib
import time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)


from sqlalchemy import func, inspect, asc, desc, exists, and_, or_, literal
from sqlalchemy.orm import Session

# ---- shared DB + models (single source of truth) ----
from db_core import engine, SessionLocal, get_db, init_schema
from models import Base, Round, Poll, Deposit, Payout, LiveEvent, PollStatus, RoundStatus
from settlement import router as settlement_router
from chain_utils import (
    verify_native_deposit,     
    to_wei_bnb,
    TREASURY_ADDRESS,
    OnChainVerifyError,
)
from aster_scoring import fetch_rankings


from aster_price import (
    get_live_price,
    previous_candle_color,
    pct_change_over_window,
    live_mini_card,
    candle_open_close_snapshot,
)


TEMPLATE_MAP = {
    "NEXT_5M_CANDLE_COLOR": ("CANDLE_COLOR", "5m", 300, 300),
    "NEXT_1M_CANDLE_COLOR": ("CANDLE_COLOR", "1m", 60, 60),
    "PRICE_AFTER_5M_VS_NOW": ("PRICE_UP_DOWN", "5m", 300, 300),
}

DEFAULT_TEMPLATES = [
    "NEXT_5M_CANDLE_COLOR",
    "PRICE_AFTER_5M_VS_NOW",
    "NEXT_1M_CANDLE_COLOR",
]

ACTIVE_STATUSES   = [PollStatus.OPEN, PollStatus.DEPOSIT_CLOSED, PollStatus.SETTLING]
INACTIVE_STATUSES = [PollStatus.SETTLED]


MIN_BET_BNB = Decimal(os.getenv("MIN_BET_BNB") or "0.00001")
MIN_CONFS   = int(os.getenv("CONFIRMATIONS") or 1)

Base.metadata.create_all(bind=engine)

def _log_db_state():
    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL") or "sqlite:///./aster_predict.db"
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        print("[DB] Using:", os.path.abspath(db_path))
    insp = inspect(engine)
    print("[DB] Tables:", insp.get_table_names())

_log_db_state()


MODULE_NAME = "poll_autopilot"       


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
# ---------- FastAPI ----------
app = FastAPI(title="4cast API – Full")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(settlement_router) 
# ---------- tiny helpers ----------
def now_ms() -> int:
    return int(time.time() * 1000)

def ms_remaining(to_ms: int) -> int:
    return max(0, to_ms - now_ms())

def jloads(s: Optional[str]) -> Optional[Dict[str, Any]]:
    return json.loads(s) if s else None

def jdumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))

def add_live(db: Session, message: str, ctx: Optional[Dict[str, Any]] = None):
    db.add(LiveEvent(kind="TEXT", message=message, ctx_json=jdumps(ctx) if ctx else None))
    db.commit()

def gen_question(symbol: str, kind: str, label: str) -> str:
    sym = symbol.upper()
    if kind == "CANDLE_COLOR":
        frame = "1-minute" if label == "1m" else "5-minute"
        return f"Will the next {frame} {sym} candle close GREEN or RED?"
    else:
        frame = "1 minute" if label == "1m" else "5 minutes"
        return f"Will {sym} price be higher or lower in {frame}?"

def side_list(kind: str) -> List[str]:
    return ["GREEN", "RED"] if kind == "CANDLE_COLOR" else ["UP", "DOWN"]

def decide_result(kind: str, open_price: float, close_price: float) -> str:
    if kind == "CANDLE_COLOR":
        return "GREEN" if close_price >= open_price else "RED"
    else:
        return "UP" if close_price >= open_price else "DOWN"

def send_chain_transfer(to_addr: str, amount_bnb: float) -> str:
    return "0x" + uuid.uuid4().hex 

def get_or_create_latest_round_id(db: Session) -> int:
    rid = db.query(Round.id).order_by(Round.id.desc()).scalar()
    if rid is not None:
        return rid
    r = Round(status=RoundStatus.LIVE, note="bootstrap")
    db.add(r)
    db.commit()
    db.refresh(r)
    return r.id




def short_addr(addr: str) -> str:
    if not addr or len(addr) < 10:
        return addr or ""
    return f"{addr[:6]}…{addr[-4:]}" 

def compute_results_block(db: Session, p: Poll) -> Optional[Dict[str, Any]]:
    if p.status != PollStatus.SETTLED or not p.winner_side:
        return None

    rows: List[Deposit] = db.query(Deposit).filter(Deposit.poll_id == p.id).all()
    total_pot = float(sum(d.amount_bnb for d in rows))
    fee_bnb = round((p.fee_bps / 10000.0) * total_pot, 8)
    distributable = max(0.0, total_pot - fee_bnb)

    winners = [d for d in rows if d.side == p.winner_side]
    winners_sum = float(sum(w.amount_bnb for w in winners)) or 0.0

    participants = db.query(func.count(func.distinct(Deposit.user_address)))\
                     .filter(Deposit.poll_id == p.id).scalar() or 0

    winner_items = []
    for w in winners:
        stake = float(w.amount_bnb)
        share = (stake / winners_sum) if winners_sum > 0 else 0.0
        payout = round(share * distributable, 8)
        profit = round(payout - stake, 8)
        roi_pct = round((profit / stake) * 100.0, 2) if stake > 0 else 0.0
        winner_items.append({
            "user_address": w.user_address,
            "user_short": short_addr(w.user_address),
            "bet_bnb": round(stake, 8),
            "payout_bnb": payout,
            "profit_bnb": profit,          
            "roi_pct": roi_pct,           
            "share_pct": round(share * 100.0, 4),
        })

    winner_items.sort(key=lambda x: x["payout_bnb"], reverse=True)
    for i, item in enumerate(winner_items, start=1):
        item["rank"] = i

    return {
        "summary": {
            "winner_side": p.winner_side,
            "total_pot_bnb": round(total_pot, 8),
            "house_fee_pct": round(p.fee_bps / 100.0, 2),
            "house_fee_bnb": round(fee_bnb, 8),
            "participants": int(participants),
            "winners_count": len(winner_items),
        },
        "winners": winner_items
    }


# ---------- serializers ----------
class PollCard(BaseModel):
    id: int
    symbol: str
    question: str
    status: PollStatus
    template: Dict[str, Any]
    timers: Dict[str, Any]
    market: Dict[str, Any]
    sides: Dict[str, Any]
    pot: Dict[str, Any]
    snapshots: Dict[str, Any]
    results: Optional[PollResults] = None 

    class Config:
        from_attributes = True

class PollSummary(BaseModel):
    id: int
    symbol: str
    question: str
    status: PollStatus
    template: Dict[str, Any]

    class Config:
        from_attributes = True


# ----- Results DTOs for "View Results" modal -----
class PollResultsSummary(BaseModel):
    winner_side: str
    total_pot_bnb: float
    house_fee_pct: float
    house_fee_bnb: float
    participants: int
    winners_count: int

class PollWinnerItem(BaseModel):
    rank: int
    user_address: str
    user_short: str
    bet_bnb: float
    payout_bnb: float
    profit_bnb: float
    roi_pct: float
    share_pct: float

class PollResults(BaseModel):
    summary: PollResultsSummary
    winners: List[PollWinnerItem]

class StartRoundsIn(BaseModel):
    symbols_count: int = Field(3, ge=1, le=10)
    templates: List[str] = Field(default_factory=lambda: DEFAULT_TEMPLATES)
    fee_bps: int = Field(200, ge=0, le=10000)
    deposit_window_seconds: Optional[int] = None
    prediction_window_seconds: Optional[int] = None
    symbols: Optional[List[str]] = None
    exclude_symbols: Optional[List[str]] = None

class StartRoundsOut(BaseModel):
    round_id: int
    created: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]
    


def get_or_create_latest_round_id(db: Session) -> int:
    r = db.query(Round).order_by(Round.id.desc()).first()
    if r:
        return r.id
    r = Round(status=RoundStatus.LIVE, note="auto")
    db.add(r)
    db.commit()
    return r.id

def _has_conflict(db: Session, symbol: str, kind: str, label: str) -> bool:
    live_statuses = [PollStatus.OPEN, PollStatus.DEPOSIT_CLOSED, PollStatus.SETTLING]
    exists = db.query(Poll.id).filter(
        Poll.symbol == symbol.upper(),
        Poll.template_kind == kind,
        Poll.template_label == label,
        Poll.status.in_(live_statuses),
    ).first()
    return bool(exists)

def _seconds_for_label(label: str) -> int:
    return 300 if label == "5m" else 60



    
    
async def to_card(p: Poll, db: Session) -> PollCard:
    if p.status == PollStatus.OPEN:
        title = "Deposit closes"
        timer_to = p.deposit_close_ms
    elif p.status == PollStatus.DEPOSIT_CLOSED:
        title = "Prediction closes"
        timer_to = p.target_close_ms
    elif p.status == PollStatus.SETTLING:
        title = "Settling"
        timer_to = p.target_close_ms
    else:
        title = "Settled"
        timer_to = now_ms()

    
    if p.template_kind == "CANDLE_COLOR":
        sides = ["GREEN", "RED"]
    else:
        sides = ["UP", "DOWN"]

    sums = (
        db.query(Deposit.side, func.coalesce(func.sum(Deposit.amount_bnb), 0.0))
          .filter(Deposit.poll_id == p.id, Deposit.side.in_(sides))
          .group_by(Deposit.side)
          .all()
    )
    totals = {s: 0.0 for s in sides}
    for side, amt in sums:
        totals[side] = float(amt or 0.0)
    total_bnb = sum(totals.values())
    shares = {s: (totals[s] / total_bnb) if total_bnb > 0 else 0.0 for s in sides}

    if p.template_kind == "CANDLE_COLOR":
        sides_block = {
            "GREEN": {"percent": shares["GREEN"], "amount_bnb": round(totals["GREEN"], 4)},
            "RED":   {"percent": shares["RED"],   "amount_bnb": round(totals["RED"], 4)},
        }
    else:
        sides_block = {
            "UP":   {"percent": shares["UP"],   "amount_bnb": round(totals["UP"], 4)},
            "DOWN": {"percent": shares["DOWN"], "amount_bnb": round(totals["DOWN"], 4)},
        }

    
    results_block = compute_results_block(db, p) if p.status == PollStatus.SETTLED else None

    res = PollCard(
        id=p.id,
        symbol=p.symbol,
        question=(p.question or gen_question(p.symbol, p.template_kind, p.template_label)),
        status=p.status,
        template={
            "label": p.template_label,
            "kind": p.template_kind,
            "sides": side_list(p.template_kind),
            "fee_bps": p.fee_bps
        },
        timers={
            "now_ms": now_ms(),
            "timer_to_ms": timer_to,
            "remaining_ms": ms_remaining(timer_to),
            "title": title
        },
        market={
            "live_price": get_live_price(p.symbol),
            "change_5m_pct": pct_change_over_window(p.symbol, '5m'),
            "previous_candle": previous_candle_color(p.symbol, '5m'),
            "mark_source": "Aster DEX",
        },
        sides=sides_block,
        pot={
            "total_bnb": round(total_bnb, 4), 
            "participants": p.participants,
            "fee_percent": p.fee_bps / 100.0
        },
        snapshots={
            "open": jloads(p.open_snapshot_json),
            "close": jloads(p.close_snapshot_json),
            "winner_side": p.winner_side
        },
       
        results=results_block
    )
    return res



def to_summary(p: Poll) -> PollSummary:
    return PollSummary(
        id=p.id,
        symbol=p.symbol,
        question=(p.question or gen_question(p.symbol, p.template_kind, p.template_label)),
        status=p.status,
        template={"label": p.template_label, "kind": p.template_kind}
    )

# ---------- seed ----------
def seed(db: Session):
    if db.query(Round).count() == 0:
        db.add(Round(status=RoundStatus.LIVE, note="bootstrap"))
        db.commit()
    if db.query(Poll).count() == 0:
        base = now_ms()
        def mk(symbol, lbl, kind, pot, ppl):
            dep_ms = 5*60*1000 if lbl=="5m" else 60*1000
            p = Poll(
                round_id=db.query(Round.id).first()[0],
                symbol=symbol,
                status=PollStatus.OPEN,
                template_kind=kind,
                template_label=lbl,
                deposit_open_ms=base,
                deposit_close_ms=base + dep_ms,
                target_open_ms=base + dep_ms,
                target_close_ms=base + dep_ms + (5*60*1000 if lbl=="5m" else 60*1000),
                fee_bps=200,
                ref_price=get_live_price(symbol),
                pot_bnb=pot,
                participants=ppl,
                question=gen_question(symbol, kind, lbl)
            )
            db.add(p)
        mk("SOLUSDT","5m","CANDLE_COLOR", 2.8, 54)
        mk("BTCUSDT","5m","PRICE_UP_DOWN", 4.2, 91)
        mk("ETHUSDT","1m","CANDLE_COLOR", 1.5, 41)
        db.commit()

@app.on_event("startup")
def _init():
    Base.metadata.create_all(bind=engine)
    init_schema() 
        
@app.on_event("startup")
async def _start_autopilot():
    try:
        mod = importlib.import_module(MODULE_NAME)
        loop_fn = getattr(mod, "autopilot_loop", None)
        if loop_fn is None:
            print(f"[app] autopilot import ok, but '{MODULE_NAME}.autopilot_loop' missing")
            return
        if not asyncio.iscoroutinefunction(loop_fn):
            print(f"[app] '{MODULE_NAME}.autopilot_loop' is not async; please define 'async def autopilot_loop()'")
            return
       
        if getattr(app.state, "autopilot_task", None) and not app.state.autopilot_task.done():
            print("[app] autopilot already running; skipping")
            return
        app.state.autopilot_task = asyncio.create_task(loop_fn())
        print("[app] autopilot started")
    except Exception as e:
        print("[app] autopilot import failed:", e)

@app.on_event("shutdown")
async def _stop_autopilot():
    t = getattr(app.state, "autopilot_task", None)
    if t:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


# ---------- PUBLIC: scoring ----------
@app.get("/scoring/rankings")
def scoring_rankings():
    return {"items": fetch_rankings(None)} 

@app.get("/scoring/top")
def scoring_top(limit: int = 5):
    items = fetch_rankings(None)
    limit = max(1, min(limit, len(items))) if items else 0
    return {"items": items[:limit]}

# ---------- PUBLIC: polls ----------
def _col_exists(attr_name: str) -> bool:
    return hasattr(Poll, attr_name) and getattr(Poll, attr_name) is not None

def _col(model, name: str):
    return getattr(model, name, None)

@app.get("/polls")
async def list_polls(
    view: str = Query("card", pattern="^(list|card)$"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    q: Optional[str] = None,
    activity: Optional[str] = Query(None, pattern="^(active|inactive|all)$"),
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    template_kind: Optional[str] = None, 
    template_label: Optional[str] = None,
    round_id: Optional[int] = None,

    
    my_only: bool = Query(False),
    user_address: Optional[str] = Query(None, description="wallet address to match"),

    sort: str = Query("created_at"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    # -------- build a single filter list ----------
    filters = []
    if q:
        filters.append(Poll.symbol.ilike(f"%{q}%"))
    if symbol:
        filters.append(Poll.symbol == symbol.upper())
    if template_kind:
        filters.append(Poll.template_kind == template_kind)
    if template_label:
        filters.append(Poll.template_label == template_label)
    if round_id:
        filters.append(Poll.round_id == round_id)
    if status:
        try:
            filters.append(Poll.status == PollStatus[status])
        except KeyError:
            raise HTTPException(400, f"unknown status '{status}'")
    elif activity:
        if activity == "active":
            filters.append(Poll.status.in_(ACTIVE_STATUSES))
        elif activity == "inactive":
            filters.append(Poll.status.in_(INACTIVE_STATUSES))


    if my_only:
        if not user_address:
            raise HTTPException(400, "user_address required when my_only=true")
        addr_lc = user_address.lower()
        filters.append(
            exists().where(
                and_(
                    Deposit.poll_id == Poll.id,
                    func.lower(Deposit.user_address) == addr_lc,
                )
            )
        )

    # -------- validate sort column ----------
    colmap = {
        "created_at": _col(Poll, "created_at"),
        "id": _col(Poll, "id"),
        "deposit_close_ms": _col(Poll, "deposit_close_ms"),
        "target_close_ms": _col(Poll, "target_close_ms"),
    }
    sort_col = colmap.get(sort) or colmap.get("created_at") or colmap.get("id")
    sort_expr = (desc(sort_col) if order == "desc" else asc(sort_col)) if sort_col is not None else None

    # -------- total (no ORDER BY in count) ----------
    total = db.query(func.count(Poll.id)).filter(*filters).scalar() or 0

    # -------- page window ----------
    offset = (page - 1) * per_page
    qset = db.query(Poll).filter(*filters)
    if sort_expr is not None:
        qset = qset.order_by(sort_expr)
    rows = qset.offset(offset).limit(per_page).all()

    # -------- shape ----------
    if view == "card":
        items = await asyncio.gather(*[to_card(p, db) for p in rows])
    else:
        items = [
            {
                "id": p.id,
                "symbol": p.symbol,
                "status": p.status,
                "template_kind": p.template_kind,
                "template_label": p.template_label,
                "created_at": getattr(p, "created_at", None),
                "deposit_close_ms": getattr(p, "deposit_close_ms", None),
                "target_close_ms": getattr(p, "target_close_ms", None),
            }
            for p in rows
        ]

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "items": items,
    }
    
@app.get("/polls/{poll_id}")
async def get_poll(
    poll_id: int,
    view: Literal["card","summary"] = Query("card"),
    db: Session = Depends(get_db),
):
    p = db.query(Poll).get(poll_id)
    if not p:
        raise HTTPException(404, "poll not found")

    if view == "card":
        item = await to_card(p, db)   
    else:
        item = to_summary(p)          

    return {"item": item}

# ---------- PUBLIC: place deposit (prediction) ----------
class DepositIn(BaseModel):
    poll_id: int
    user_address: str
    side: str  
    amount_bnb: float
    tx_hash: Optional[str] = None


@app.post("/deposits")
def place_deposit(payload: DepositIn, db: Session = Depends(get_db)):
    p = db.query(Poll).get(payload.poll_id)
    if not p:
        raise HTTPException(404, "poll not found")

    # status/time guards
    if p.status != PollStatus.OPEN:
        raise HTTPException(400, "deposits closed for this poll")
    if now_ms() > p.deposit_close_ms:
        raise HTTPException(400, "deposit window over")

    # side + amount checks
    valid = side_list(p.template_kind)
    if payload.side not in valid:
        raise HTTPException(400, f"invalid side; expected one of {valid}")

    amt_bnb = Decimal(str(payload.amount_bnb or 0))


    if not payload.tx_hash:
        raise HTTPException(400, "tx_hash required")


    if db.query(Deposit.id).filter(Deposit.tx_hash == payload.tx_hash).first():
        raise HTTPException(400, "tx_hash already used")

    if not TREASURY_ADDRESS:
        raise HTTPException(500, "TREASURY_ADDRESS not configured")

    # ---- on-chain verification (native BNB) ----
    try:
        info = verify_native_deposit(
            tx_hash=payload.tx_hash,
            expected_to=TREASURY_ADDRESS,
            expected_amount_wei=to_wei_bnb(amt_bnb),
            wait_for_confirmations=True,    
        )
    except OnChainVerifyError as e:
        msg = str(e)
       
        if "pending" in msg.lower() or "not found" in msg.lower():
            raise HTTPException(202, msg)
        raise HTTPException(400, f"on-chain verification failed: {msg}")
    except Exception as e:
        raise HTTPException(400, f"on-chain verification failed: {e}")


    onchain_amount = float(info.get("value_bnb") or amt_bnb) 
    if onchain_amount <= 0:
        raise HTTPException(400, "invalid on-chain amount")


    d = Deposit(
        poll_id=p.id,
        user_address=payload.user_address,
        side=payload.side,
        amount_bnb=onchain_amount,
        tx_hash=info["hash"],
    )
    db.add(d)

  
    p.pot_bnb = float(p.pot_bnb) + onchain_amount
    p.participants = int(p.participants) + 1

    db.commit()

    add_live(
        db,
        f"Deposit verified • {payload.user_address} -> {onchain_amount} BNB on {payload.side}",
        {
            "poll_id": p.id,
            "tx_hash": info["hash"],
            "block": info.get("block_number"),
            "confirmations": info.get("confirmations", 0),
        },
    )
    return {
        "ok": True,
        "deposit_id": d.id,
        "amount_bnb": onchain_amount,
        "tx": info,
    }






# ---------- MY TRANSACTIONS ----------

def _lc(s: str) -> str:
    return (s or "").lower()

@app.get("/my/transactions")
def my_transactions(
    user_address: str = Query(..., description="wallet address (0x...)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str = Query("all", pattern="^(all|active|won|lost)$"),
    symbol: Optional[str] = None,
    order: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    """
    Returns summary (total_won, total_lost, net) and a paginated list of the user's deposits
    with computed outcome (Won/Lost/Active) and payout (if any).
    """

    addr_lc = _lc(user_address)

    base = (
        db.query(Deposit, Poll)
        .join(Poll, Poll.id == Deposit.poll_id)
        .filter(func.lower(Deposit.user_address) == addr_lc)
    )

    if symbol:
        base = base.filter(Poll.symbol == symbol.upper())

    if status == "active":
        base = base.filter(Poll.status.in_(ACTIVE_STATUSES))
    elif status == "won":
        base = base.filter(
            and_(Poll.status == PollStatus.SETTLED, Deposit.side == Poll.winner_side)
        )
    elif status == "lost":
        base = base.filter(
            and_(Poll.status == PollStatus.SETTLED, Deposit.side != Poll.winner_side)
        )

    total = base.with_entities(func.count(Deposit.id)).scalar() or 0

    order_col = Deposit.created_at
    base = base.order_by(desc(order_col) if order == "desc" else asc(order_col))


    offset = (page - 1) * per_page
    rows = base.offset(offset).limit(per_page).all()


    poll_ids = [p.id for (_, p) in rows] or [-1]
    payouts_by_poll: Dict[int, float] = dict(
        db.query(Payout.poll_id, func.coalesce(func.sum(Payout.amount_bnb), 0.0))
        .filter(
            Payout.poll_id.in_(poll_ids),
            func.lower(Payout.user_address) == addr_lc,
            Payout.is_house_fee == False,
        )
        .group_by(Payout.poll_id)
        .all()
    )


    items = []
    for d, p in rows:
        payout_bnb = float(payouts_by_poll.get(p.id, 0.0)) if p.status == PollStatus.SETTLED else 0.0

        if p.status in ACTIVE_STATUSES:
            outcome = "Active"
        elif d.side == p.winner_side and payout_bnb > 0:
            outcome = "Won"
        else:
            outcome = "Lost"

        items.append({
            "poll_id": p.id,
            "symbol": p.symbol,
            "side": d.side,
            "bet_bnb": float(d.amount_bnb),
            "payout_bnb": payout_bnb,
            "outcome": outcome,                 
            "poll_status": p.status,            
            "winner_side": p.winner_side,       
            "created_at": d.created_at.isoformat() + "Z",
            "tx_hash": d.tx_hash,
        })

    total_won = float(
        db.query(func.coalesce(func.sum(Payout.amount_bnb), 0.0))
        .filter(
            func.lower(Payout.user_address) == addr_lc,
            Payout.is_house_fee == False,
        )
        .scalar()
        or 0.0
    )


    total_lost = float(
        db.query(func.coalesce(func.sum(Deposit.amount_bnb), 0.0))
        .join(Poll, Poll.id == Deposit.poll_id)
        .filter(
            func.lower(Deposit.user_address) == addr_lc,
            Poll.status == PollStatus.SETTLED,
            Deposit.side != Poll.winner_side,
        )
        .scalar()
        or 0.0
    )

    result = {
        "summary": {
            "total_won_bnb": round(total_won, 8),
            "total_lost_bnb": round(total_lost, 8),
            "net_bnb": round(total_won - total_lost, 8),
        },
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "items": items,
    }
    return result



@app.get("/live/process/current")
def live_process_current(
    round_id: Optional[int] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    sub = (
        db.query(
            func.json_extract(LiveEvent.ctx_json, "$.poll_id").label("poll_id"),
            func.max(LiveEvent.id).label("max_id"),
        )
        .filter(LiveEvent.kind == "PROCESS")
        .group_by(func.json_extract(LiveEvent.ctx_json, "$.poll_id"))
    )

    if round_id:
        sub = sub.filter(
            func.json_extract(LiveEvent.ctx_json, "$.round_id") == round_id
        )

    sub = sub.subquery()

    rows = (
        db.query(LiveEvent)
          .join(sub, LiveEvent.id == sub.c.max_id)
          .order_by(LiveEvent.id.desc())
          .limit(limit)
          .all()
    )

    items = []
    for ev in rows:
        ctx = jloads(ev.ctx_json) or {}
        stage_id = ctx.get("stage_id")
        progress = ctx.get("progress")
        if progress is None:
            progress = _stage_percent(stage_id)

        items.append({
            "poll_id": ctx.get("poll_id"),
            "round_id": ctx.get("round_id"),
            "symbol": ctx.get("symbol"),
            "stage_id": stage_id,
            "stage_label": ctx.get("stage_label"),
            "status": ctx.get("status"),
            "progress": float(progress),
            "ts": ev.created_at.isoformat() + "Z",
            "message": ev.message,
        })

    return {"items": items}
# ---------- ADMIN: rounds ----------
class StartRoundIn(BaseModel):
    note: Optional[str] = "manual"

@app.post("/admin/rounds/start", response_model=StartRoundsOut)
def start_rounds(payload: StartRoundsIn, db: Session = Depends(get_db)):
    if payload.symbols and len(payload.symbols) > 0:
        symbols = [s.upper() for s in payload.symbols]
    else:
        from aster_scoring import fetch_rankings
        ranked = fetch_rankings(None)  # all
        if not ranked:
            raise HTTPException(503, "scoring unavailable")
        symbols = [it["symbol"].upper() for it in ranked[:payload.symbols_count]]

    if payload.exclude_symbols:
        excl = {s.upper() for s in payload.exclude_symbols}
        symbols = [s for s in symbols if s not in excl]

    round_id = get_or_create_latest_round_id(db)

    created, skipped = [], []
    base_ms = now_ms()

    for tpl_name in payload.templates:
        if tpl_name not in TEMPLATE_MAP:
            skipped.append({"template": tpl_name, "reason": "unknown_template"})
            continue

        kind, label, def_dep, def_pred = TEMPLATE_MAP[tpl_name]
        dep_secs = payload.deposit_window_seconds or def_dep or _seconds_for_label(label)
        pred_secs = payload.prediction_window_seconds or def_pred or _seconds_for_label(label)

        for sym in symbols:
            if _has_conflict(db, sym, kind, label):
                skipped.append({"symbol": sym, "template": tpl_name, "reason": "conflict_exists"})
                continue

            p = Poll(
                round_id=round_id,
                symbol=sym,
                status=PollStatus.OPEN,
                template_kind=kind,
                template_label=label,
                deposit_open_ms=base_ms,
                deposit_close_ms=base_ms + dep_secs * 1000,
                target_open_ms=base_ms + dep_secs * 1000,
                target_close_ms=base_ms + (dep_secs + pred_secs) * 1000,
                fee_bps=payload.fee_bps,
                ref_price=get_live_price(sym),
                question=gen_question(sym, kind, label),
                pot_bnb=0.0,
                participants=0,
            )
            db.add(p)
            db.commit()

            created.append({
                "poll_id": p.id,
                "symbol": sym,
                "template": tpl_name,
                "kind": kind,
                "label": label,
                "deposit_window_seconds": dep_secs,
                "prediction_window_seconds": pred_secs,
                "deposit_open_ms": p.deposit_open_ms,
                "deposit_close_ms": p.deposit_close_ms,
                "target_close_ms": p.target_close_ms,
            })

    if created:
        add_live(db, f"Started {len(created)} polls across {len(symbols)} symbol(s)", {"round_id": round_id})

    return StartRoundsOut(round_id=round_id, created=created, skipped=skipped)

# ---------- ADMIN: create poll ----------
class CreatePollIn(BaseModel):
    symbol: str
    template_kind: Literal["CANDLE_COLOR","PRICE_UP_DOWN"]
    template_label: Literal["1m","5m"]
    deposit_window_seconds: Optional[int] = None  
    prediction_window_seconds: Optional[int] = None
    fee_bps: int = 200
    round_id: Optional[int] = None

@app.post("/admin/polls")
def create_poll(body: CreatePollIn, db: Session = Depends(get_db)):
    dep_secs  = int(body.deposit_window_seconds or (300 if body.template_label == "5m" else 60))
    pred_secs = int(body.prediction_window_seconds or (300 if body.template_label == "5m" else 60))

    if dep_secs <= 0 or pred_secs <= 0:
        raise HTTPException(status_code=400, detail="deposit_window_seconds and prediction_window_seconds must be > 0")

    base_ms = now_ms()
    dep_open_ms   = base_ms
    dep_close_ms  = base_ms + dep_secs * 1000
    target_open_ms = dep_close_ms
    target_close_ms = target_open_ms + pred_secs * 1000

    if not (dep_open_ms < dep_close_ms <= target_open_ms < target_close_ms):
        raise HTTPException(status_code=400, detail="Invalid time window ordering")

    r_id = body.round_id or get_or_create_latest_round_id(db)

    p = Poll(
        round_id=r_id,
        symbol=body.symbol.upper(),
        status=PollStatus.OPEN,
        template_kind=body.template_kind,
        template_label=body.template_label,

        deposit_open_ms=dep_open_ms,
        deposit_close_ms=dep_close_ms,
        target_open_ms=target_open_ms,
        target_close_ms=target_close_ms,

        fee_bps=body.fee_bps,
        ref_price=get_live_price(body.symbol),
        question=gen_question(body.symbol, body.template_kind, body.template_label),
        pot_bnb=0.0,
        participants=0,
    )

    db.add(p)
    db.commit()


    add_live(
        db,
        f"Poll created • {p.symbol} {p.template_label} {p.template_kind}",
        {
            "poll_id": p.id,
            "deposit_open_ms": dep_open_ms,
            "deposit_close_ms": dep_close_ms,
            "target_open_ms": target_open_ms,
            "target_close_ms": target_close_ms,
        },
    )

    return {"ok": True, "poll_id": p.id}


# ---------- ADMIN: force close deposits (take opening snapshot) ----------
@app.post("/admin/polls/{poll_id}/close-deposits")
def force_close_deposits(poll_id: int, db: Session = Depends(get_db)):
    p = db.query(Poll).get(poll_id)
    if not p:
        raise HTTPException(404, "poll not found")
    if p.status != PollStatus.OPEN:
        raise HTTPException(400, "poll is not OPEN")

    p.status = PollStatus.DEPOSIT_CLOSED
    open_px = get_live_price(p.symbol)
    p.open_snapshot_json = jdumps({
        "open_time_ms": now_ms(),
        "open_price": open_px
    })
    if p.ref_price is None:
        p.ref_price = open_px
    db.commit()
    add_live(db, f"Deposits closed • opening snapshot taken ({open_px})", {"poll_id": p.id})
    return {"ok": True}

# ---------- SETTLEMENT helpers ----------
def take_open_snapshot_if_needed(p: Poll):
    if not p.open_snapshot_json:
        p.open_snapshot_json = jdumps({
            "open_time_ms": p.target_open_ms,
            "open_price": get_live_price(p.symbol)
        })
        if p.ref_price is None:
            p.ref_price = json.loads(p.open_snapshot_json)["open_price"]

def take_close_snapshot(p: Poll):
    p.close_snapshot_json = jdumps({
        "close_time_ms": now_ms(),
        "close_price": get_live_price(p.symbol)
    })

def compute_and_payout(db: Session, p: Poll):
    take_open_snapshot_if_needed(p)
    take_close_snapshot(p)

    open_px = json.loads(p.open_snapshot_json)["open_price"]
    close_px = json.loads(p.close_snapshot_json)["close_price"]
    winner = decide_result(p.template_kind, open_px, close_px)

    rows = db.query(Deposit).filter(Deposit.poll_id == p.id).all()
    total = sum(d.amount_bnb for d in rows)
    fee = (p.fee_bps / 10000.0) * total
    winners = [d for d in rows if d.side == winner]
    losers = [d for d in rows if d.side != winner]

    if fee > 0:
        db.add(Payout(
            poll_id=p.id, user_address="HOUSE_TREASURY", amount_bnb=round(fee, 8), is_house_fee=True,
            tx_hash=send_chain_transfer("HOUSE_TREASURY", fee)
        ))

    distributable = max(0.0, total - fee)
    if winners and distributable > 0:
        winners_sum = sum(w.amount_bnb for w in winners)
        for w in winners:
            share = (w.amount_bnb / winners_sum) * distributable
            db.add(Payout(
                poll_id=p.id, user_address=w.user_address, amount_bnb=round(share, 8),
                is_house_fee=False, tx_hash=send_chain_transfer(w.user_address, share)
            ))

    p.winner_side = winner
    p.status = PollStatus.SETTLED
    p.settled_at_ms = now_ms()
    db.commit()

    add_live(db, f"Poll settled • winner: {winner} • pot: {round(total,4)} BNB", {"poll_id": p.id})

# ---------- ADMIN: settle ready (sweep) ----------
@app.post("/admin/settle/ready")
def settle_ready(db: Session = Depends(get_db)):
    now = now_ms()
    candidates = db.query(Poll).filter(
        Poll.target_close_ms <= now,
        Poll.status.in_([PollStatus.OPEN, PollStatus.DEPOSIT_CLOSED, PollStatus.SETTLING])
    ).all()
    settled = []
    for p in candidates:
        if p.status == PollStatus.OPEN:
            p.status = PollStatus.DEPOSIT_CLOSED
        p.status = PollStatus.SETTLING
        db.commit()
        compute_and_payout(db, p)
        settled.append(p.id)
    return {"ok": True, "settled_ids": settled, "count": len(settled)}


from decimal import Decimal, InvalidOperation
from sqlalchemy import func
from datetime import datetime, timezone

def _bnb_fmt(x, dp: int = 8) -> str:
    try:
        d = Decimal(str(x))
        q = d.quantize(Decimal(1).scaleb(-dp))
        s = format(q, 'f').rstrip('0').rstrip('.')
        return s or '0'
    except (InvalidOperation, ValueError, TypeError):
        return str(x)

def _iso(ms: int | None):
    if not ms:
        return None
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()

def _dur(ms_start: int | None, ms_end: int | None):
    if not ms_start or not ms_end:
        return None
    sec = max(0, int((ms_end - ms_start)/1000))
    if sec < 60:  return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:    return f"{m}m {s}s"
    h, r = divmod(m, 60)
    return f"{h}h {r}m"

def _load_poll(db: Session, poll_id: int) -> Optional[Poll]:
    return db.query(Poll).get(poll_id)

def _pot_and_participants(db: Session, poll_id: int) -> tuple[float, int]:
    p = _load_poll(db, poll_id)
    if not p:
        return 0.0, 0
    pot = float(p.pot_bnb or 0.0)
    participants = int(p.participants or 0)
    if pot <= 0.0 or participants == 0:
        pot = float(
            db.query(func.coalesce(func.sum(Deposit.amount_bnb), 0.0))
              .filter(Deposit.poll_id == poll_id).scalar() or 0.0
        )
        participants = int(
            db.query(func.count(func.distinct(Deposit.user_address)))
              .filter(Deposit.poll_id == poll_id).scalar() or 0
        )
    return pot, participants

def _winner_stats(db: Session, poll_id: int) -> tuple[int, float]:
    row = (db.query(
                func.count(func.distinct(Payout.user_address)),
                func.coalesce(func.sum(Payout.amount_bnb), 0.0)
            )
            .filter(Payout.poll_id == poll_id, Payout.is_house_fee == False)
            .first() or (0, 0.0))
    return int(row[0] or 0), float(row[1] or 0.0)

def _losers_count(db: Session, poll: Poll) -> int:
    if not poll or poll.status != PollStatus.SETTLED or not poll.winner_side:
        return 0
    return int(
        (db.query(func.count(func.distinct(Deposit.user_address)))
           .filter(Deposit.poll_id == poll.id, Deposit.side != poll.winner_side)
           .scalar()) or 0
    )

def _snap_prices(p: Poll) -> tuple[float|None, float|None]:
    try:
        o = (jloads(p.open_snapshot_json) or {}).get("open_price")
        c = (jloads(p.close_snapshot_json) or {}).get("close_price")
        return (float(o) if o is not None else None,
                float(c) if c is not None else None)
    except Exception:
        return (None, None)

def _pct(a: float|None, b: float|None) -> str|None:
    if a is None or b is None or a == 0:
        return None
    return f"{((b-a)/a)*100:.2f}%"

def _fee_amount(p: Poll, pot_bnb: float) -> float:
    try:
        return float(p.fee_bps or 0) / 10000.0 * float(pot_bnb)
    except Exception:
        return 0.0

def _details_lines_for_event(db: Session, ev: LiveEvent, ctx: dict) -> list[str]:
    out: list[str] = []
    pid = ctx.get("poll_id")
    if not pid:
        return out

    p = _load_poll(db, int(pid))
    if not p:
        return out

    pot_bnb, participants = _pot_and_participants(db, p.id)
    winners_count, winning_pool_bnb = _winner_stats(db, p.id)
    fee_bnb = _fee_amount(p, pot_bnb)
    distributable_bnb = max(0.0, pot_bnb - fee_bnb)
    losers_count = _losers_count(db, p)
    o_px, c_px = _snap_prices(p)
    delta_pct = _pct(o_px, c_px)
    dep_open_iso = _iso(p.deposit_open_ms)
    dep_close_iso = _iso(p.deposit_close_ms)
    tar_open_iso = _iso(p.target_open_ms)
    tar_close_iso = _iso(p.target_close_ms)
    dep_dur = _dur(p.deposit_open_ms, p.deposit_close_ms)
    pred_dur = _dur(p.target_open_ms, p.target_close_ms)

    mlow = (ev.message or "").lower()
    stage = (ctx.get("stage_id") or "").lower()

    if "auto-started poll" in mlow or ("poll created" in mlow) or stage in {"creating-poll", "poll-created"}:
        out.append(f"{p.symbol} • {p.template_label} • {p.template_kind}")
        out.append(f"Deposit window: {dep_open_iso} → {dep_close_iso} ({dep_dur})")
        out.append(f"Prediction window: {tar_open_iso} → {tar_close_iso} ({pred_dur})")
        out.append(f"House fee: {float(p.fee_bps)/100:.2f}%")
        return out

    if "deposits closed" in mlow or stage in {"polling-complete"}:
        out.append(f"Pot: {_bnb_fmt(pot_bnb)} BNB • Participants: {participants}")
        out.append(f"Target closes at: {tar_close_iso}")
        return out

    if "moving to settling" in mlow or "target timestamp closed" in mlow or stage in {"target-closed"}:
        base = f"Target closed at: {tar_close_iso}"
        if o_px is not None and c_px is not None:
            base += f" • Open: {o_px} • Close: {c_px}" + (f" • Δ: {delta_pct}" if delta_pct else "")
        out.append(base)
        return out

    if "result:" in mlow or "settled" in mlow or stage in {"announcing", "result-announced"}:
        out.append(f"Winner: {p.winner_side or ctx.get('winner') or '—'}")
        if o_px is not None and c_px is not None:
            line = f"Open: {o_px} • Close: {c_px}"
            if delta_pct:
                line += f" • Δ: {delta_pct}"
            out.append(line)
        out.append(f"House fee: {_bnb_fmt(fee_bnb)} BNB • Distributable: {_bnb_fmt(distributable_bnb)} BNB")
        if p.status == PollStatus.SETTLED:
            out.append(f"Winners: {winners_count} • Losers: {losers_count}")
        return out

    if "distribution in process" in mlow or stage in {"distributing"}:
        avg = winning_pool_bnb / winners_count if winners_count else 0.0
        out.append(f"Distributing {_bnb_fmt(distributable_bnb)} BNB to {winners_count} winners")
        if winners_count:
            out.append(f"Avg payout (approx): {_bnb_fmt(avg)} BNB")
        return out

    if "payouts distributed" in mlow or stage in {"distributed"}:
        avg = winning_pool_bnb / winners_count if winners_count else 0.0
        out.append(f"Total sent: {_bnb_fmt(winning_pool_bnb)} BNB • Winners: {winners_count}")
        if winners_count:
            out.append(f"Avg payout: {_bnb_fmt(avg)} BNB • Losers: {losers_count}")
        return out


    if "result verification" in mlow or stage in {"result-process", "calculating", "winners-calc"}:
        out.append(f"Pot: {_bnb_fmt(pot_bnb)} BNB • Fee: {float(p.fee_bps)/100:.2f}% • Distributable: {_bnb_fmt(distributable_bnb)} BNB")
        if o_px is not None:
            out.append(f"Open px: {o_px} • Close px: {c_px if c_px is not None else '—'}")
        return out


    out.append(f"Poll #{p.id} • {p.symbol} • Pot: {_bnb_fmt(pot_bnb)} BNB • Participants: {participants}")
    return out



@app.get("/live/events")
def live_events(
    kind: Optional[Literal["TEXT","PROCESS","ALL"]] = Query("ALL"),
    poll_id: Optional[int] = Query(None),
    round_id: Optional[int] = Query(None),


    since_id: Optional[int] = Query(None, description="return events with id > since_id"),
    after_ts: Optional[int]  = Query(None, description="ms since epoch (created_at >=)"),
    before_ts: Optional[int] = Query(None, description="ms since epoch (created_at <=)"),


    limit: int = Query(100, ge=1, le=500),
    order: Literal["asc","desc"] = Query("asc"),

    db: Session = Depends(get_db),
):
    q = db.query(LiveEvent)

    if kind and kind != "ALL":
        q = q.filter(LiveEvent.kind == kind)

    if poll_id is not None:
        q = q.filter(func.json_extract(LiveEvent.ctx_json, "$.poll_id") == poll_id)
    if round_id is not None:
        q = q.filter(func.json_extract(LiveEvent.ctx_json, "$.round_id") == round_id)

    if since_id is not None:
        q = q.filter(LiveEvent.id > since_id)
    if after_ts is not None:
        q = q.filter(func.strftime("%s", LiveEvent.created_at) * 1000 >= after_ts)
    if before_ts is not None:
        q = q.filter(func.strftime("%s", LiveEvent.created_at) * 1000 <= before_ts)

    q = q.order_by(LiveEvent.id.asc() if order == "asc" else LiveEvent.id.desc())
    rows = q.limit(limit).all()

    def serialize(ev: LiveEvent):
        ctx = jloads(ev.ctx_json) or {}
        return {
            "id": ev.id,
            "ts": int(ev.created_at.timestamp() * 1000),
            "kind": ev.kind,
            "message": ev.message,                    
            "ctx": ctx,
            "details": _details_lines_for_event(db, ev, ctx), 
        }

    items = [serialize(r) for r in rows]
    if order == "desc":
        items_rev = list(reversed(items))
    else:
        items_rev = items

    next_since_id = items[-1]["id"] if items else (since_id or 0)

    return {
        "items": items_rev,        
        "count": len(items_rev),
        "order": "asc",
        "next_since_id": next_since_id
    }


# ---------- Leaderboard ----------
def _short(addr: str) -> str:
    if not addr or len(addr) < 10:
        return addr or ""
    return f"{addr[:6]}...{addr[-4:]}"

def _safe_div(a: Optional[float], b: Optional[float]) -> float:
    try:
        a = float(a or 0.0)
        b = float(b or 0.0)
        return (a / b) if b > 0 else 0.0
    except Exception:
        return 0.0


@app.get("/leaderboard/summary")
def leaderboard_summary(db: Session = Depends(get_db)):
    total_traders = int(
        db.query(func.count(func.distinct(Deposit.user_address))).scalar() or 0
    )

    total_volume_bnb = float(
        db.query(func.coalesce(func.sum(Deposit.amount_bnb), 0.0)).scalar() or 0.0
    )
    sp = db.query(Poll.id, Poll.winner_side).filter(Poll.status == PollStatus.SETTLED).subquery()

    settled_trades_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.count(literal(1)).label("settled_trades"),
        )
        .join(sp, Deposit.poll_id == sp.c.id)
        .group_by(Deposit.user_address)
        .subquery()
    )

    wins_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.count(literal(1)).label("wins"),
        )
        .join(sp, Deposit.poll_id == sp.c.id)
        .filter(Deposit.side == sp.c.winner_side)
        .group_by(Deposit.user_address)
        .subquery()
    )

    rows = (
        db.query(
            settled_trades_q.c.user,
            settled_trades_q.c.settled_trades,
            func.coalesce(wins_q.c.wins, 0).label("wins"),
        )
        .outerjoin(wins_q, wins_q.c.user == settled_trades_q.c.user)
        .all()
    )

    per_user_rates = [
        _safe_div(r.wins, r.settled_trades) for r in rows if (r.settled_trades or 0) > 0
    ]
    avg_win_rate_pct = round(
        (sum(per_user_rates) / len(per_user_rates) * 100.0) if per_user_rates else 0.0, 1
    )

    return {
        "total_traders": total_traders,
        "total_volume_bnb": round(total_volume_bnb, 8),
        "avg_win_rate_pct": avg_win_rate_pct,
    }


@app.get("/leaderboard/top")
def leaderboard_top(
    db: Session = Depends(get_db),
    page: int = 1,
    per_page: int = 50,
    sort: str = "profit", 
    order: str = "desc",
):
    page = max(1, page)
    per_page = max(1, min(200, per_page))


    sp = db.query(Poll.id, Poll.winner_side).filter(Poll.status == PollStatus.SETTLED).subquery()

    trades_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.count(literal(1)).label("trades"),
        )
        .group_by(Deposit.user_address)
        .subquery()
    )

    settled_trades_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.count(literal(1)).label("settled_trades"),
        )
        .join(sp, Deposit.poll_id == sp.c.id)
        .group_by(Deposit.user_address)
        .subquery()
    )

    wins_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.count(literal(1)).label("wins"),
        )
        .join(sp, Deposit.poll_id == sp.c.id)
        .filter(Deposit.side == sp.c.winner_side)
        .group_by(Deposit.user_address)
        .subquery()
    )

    dep_sum_q = (
        db.query(
            Deposit.user_address.label("user"),
            func.coalesce(func.sum(Deposit.amount_bnb), 0.0).label("deposits_bnb"),
        )
        .group_by(Deposit.user_address)
        .subquery()
    )

    pay_sum_q = (
        db.query(
            Payout.user_address.label("user"),
            func.coalesce(func.sum(Payout.amount_bnb), 0.0).label("payouts_bnb"),
        )
        .filter(Payout.is_house_fee == False)
        .group_by(Payout.user_address)
        .subquery()
    )

    base_q = (
        db.query(
            trades_q.c.user.label("user"),
            trades_q.c.trades.label("trades"),
            func.coalesce(settled_trades_q.c.settled_trades, 0).label("settled_trades"),
            func.coalesce(wins_q.c.wins, 0).label("wins"),
            func.coalesce(dep_sum_q.c.deposits_bnb, 0.0).label("deposits_bnb"),
            func.coalesce(pay_sum_q.c.payouts_bnb, 0.0).label("payouts_bnb"),
        )
        .outerjoin(settled_trades_q, settled_trades_q.c.user == trades_q.c.user)
        .outerjoin(wins_q, wins_q.c.user == trades_q.c.user)
        .outerjoin(dep_sum_q, dep_sum_q.c.user == trades_q.c.user)
        .outerjoin(pay_sum_q, pay_sum_q.c.user == trades_q.c.user)
    )

    total = base_q.count()


    profit_expr = (func.coalesce(pay_sum_q.c.payouts_bnb, 0.0) - func.coalesce(dep_sum_q.c.deposits_bnb, 0.0))
    win_rate_expr = func.coalesce(
        wins_q.c.wins / func.nullif(settled_trades_q.c.settled_trades, 0),
        0.0
    )

    if sort == "trades":
        sort_col = trades_q.c.trades
    elif sort == "win_rate":
        sort_col = win_rate_expr
    else:
        sort = "profit"
        sort_col = profit_expr

    sort_dir = desc if order.lower() == "desc" else asc

    rows = (
        base_q
        .order_by(sort_dir(sort_col))
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    items = []
    for idx, r in enumerate(rows, start=(page - 1) * per_page + 1):
        trades = int(r.trades or 0)
        settled_trades = int(r.settled_trades or 0)
        wins = int(r.wins or 0)

        win_rate = _safe_div(wins, settled_trades) * 100.0  # %
        profit = float((r.payouts_bnb or 0.0) - (r.deposits_bnb or 0.0))

        items.append({
            "rank": idx,
            "address": r.user,
            "address_short": _short(r.user),
            "trades": trades,
            "win_rate_pct": round(win_rate, 1),
            "total_profit_bnb": round(profit, 8),
        })

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page if total else 0,
        "sort": sort,
        "order": order.lower(),
        "items": items,
    }


# ---------- HEALTH ----------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_ms()}


@app.get("/__debug__/autopilot")
def debug_autopilot():
    t = getattr(app.state, "autopilot_task", None)
    return {
        "task_exists": bool(t),
        "task_done": (t.done() if t else None),
        "module": "poll_autopilot" if t else None,
    }
