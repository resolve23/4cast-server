import os
import time
import json
import asyncio
from typing import List, Tuple
import os
from datetime import datetime, timezone

from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
load_dotenv(find_dotenv(), override=False)

from sqlalchemy.orm import Session
from sqlalchemy import func

from aster_scoring import fetch_rankings

from db_core import SessionLocal
from models import Poll, Deposit, Payout, LiveEvent, PollStatus, Round, RoundStatus
from app import now_ms, get_live_price, gen_question, add_live, get_db


from settlement import (
    send_chain_transfer,                 
    decide_result,                       
    take_open_snapshot_if_needed,        
    take_close_snapshot,                 
    TREASURY_ADDR,                       
)

# =========================
# Config / Timing Guards
# =========================
CLOCK_SKEW_MS = int(os.getenv("CLOCK_SKEW_MS", "1000"))                 
SETTLEMENT_GRACE_MS = int(os.getenv("SETTLEMENT_GRACE_MS", "1000"))     
TICK_SECONDS = int(os.getenv("AUTOPILOT_TICK_SECONDS", "15"))           
MAX_SETTLEMENTS_PER_TICK = int(os.getenv("AUTOPILOT_MAX_SETTLEMENTS", "1"))  
DEBUG_AUTOPILOT = os.getenv("DEBUG_AUTOPILOT", "1") not in ("0", "false", "False")



TEMPLATE_MAP = {
    "NEXT_5M_CANDLE_COLOR": ("CANDLE_COLOR", "5m"),
    "NEXT_1M_CANDLE_COLOR": ("CANDLE_COLOR", "1m"),
    "PRICE_AFTER_5M_VS_NOW": ("PRICE_UP_DOWN", "5m"),
}

def _env_bool(name: str, default="false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [s.strip() for s in raw.split(",") if s.strip()]

def _seconds_for_label(label: str) -> int:
    return 300 if label == "5m" else 60

def _active_count(db: Session, symbol: str, kind: str, label: str) -> int:
    live = [PollStatus.OPEN, PollStatus.DEPOSIT_CLOSED, PollStatus.SETTLING]
    return db.query(func.count(Poll.id)).filter(
        Poll.symbol == symbol.upper(),
        Poll.template_kind == kind,
        Poll.template_label == label,
        Poll.status.in_(live),
    ).scalar() or 0

def _get_or_create_latest_round_id(db: Session) -> int:
    r = db.query(Round).order_by(Round.id.desc()).first()
    if r:
        return r.id
    r = Round(status=RoundStatus.LIVE, note="auto")
    db.add(r)
    db.commit()
    return r.id

def _create_poll(
    db: Session,
    symbol: str,
    kind: str,
    label: str,
    fee_bps: int,
    dep_secs: int,
    pred_secs: int,
) -> Poll:
    base = now_ms()
    r_id = _get_or_create_latest_round_id(db)

    p = Poll(
        round_id=r_id,
        symbol=symbol.upper(),
        status=PollStatus.OPEN,
        template_kind=kind,
        template_label=label,
        deposit_open_ms=base,
        deposit_close_ms=base + dep_secs * 1000,
        target_open_ms=base + dep_secs * 1000,
        target_close_ms=base + (dep_secs + pred_secs) * 1000,
        fee_bps=fee_bps,
        ref_price=get_live_price(symbol),
        question=gen_question(symbol, kind, label),
        pot_bnb=0.0,
        participants=0,
    )
    db.add(p)
    db.commit()
    add_live(db, f"Auto-started poll • {symbol} {label} {kind}", {"poll_id": p.id})
    return p

_last_auto_seed_ts = 0.0 


def _label_override(label: str, kind: str, which: str) -> int:
    keys = [
        f"AUTOPILOT_{kind}_{label.upper()}_{which}",  
        f"AUTOPILOT_{label.upper()}_{which}",        
    ]
    for k in keys:
        v = os.getenv(k)
        if v and v.isdigit():
            return int(v)
    return 0

def autopilot_seed_rounds(db: Session):
    global _last_auto_seed_ts

    if not _env_bool("AUTOPILOT_START_ROUNDS", "false"):
        return

    now = time.time()
    cooldown = _env_int("AUTOPILOT_COOLDOWN_SEC", 10)
    if (now - _last_auto_seed_ts) < cooldown:
        return

    symbols_count = _env_int("AUTOPILOT_SYMBOLS_COUNT", 3)
    templates = _env_list(
        "AUTOPILOT_TEMPLATES",
        "NEXT_5M_CANDLE_COLOR,PRICE_AFTER_5M_VS_NOW,NEXT_1M_CANDLE_COLOR",
    )
    fee_bps = _env_int("AUTOPILOT_FEE_BPS", 200)
    dep_sec_override = _env_int("AUTOPILOT_DEPOSIT_WINDOW_SEC", 0)
    pred_sec_override = _env_int("AUTOPILOT_PREDICTION_WINDOW_SEC", 0)
    max_conc = _env_int("AUTOPILOT_MAX_CONCURRENT_PER_SYMBOL", 1)

    ranked = fetch_rankings(None) or []
    symbols = [it["symbol"].upper() for it in ranked[:symbols_count]]
    if not symbols:
        return

    created = 0
    for sym in symbols:
        for tpl in templates:
            if tpl not in TEMPLATE_MAP:
                continue
            kind, label = TEMPLATE_MAP[tpl]

           
            if _active_count(db, sym, kind, label) >= max_conc:
                continue

            d = _label_override(label, kind, "DEP") or dep_sec_override or _seconds_for_label(label)
            p = _label_override(label, kind, "PRED") or pred_sec_override or _seconds_for_label(label)

            _create_poll(db, sym, kind, label, fee_bps, d, p)
            created += 1

    if created:
        print(f"[autopilot] auto-started {created} poll(s) for {symbols}")
    _last_auto_seed_ts = now

# =========================
# Utils
# =========================
def ms_to_iso(ms: int | None) -> str:
    if not ms:
        return "None"
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()

def rel_secs(ms: int | None, now_ms: int) -> str:
    if ms is None:
        return "n/a"
    d = (ms - now_ms) / 1000.0
    sign = "" if d < 0 else "+"
    return f"{sign}{d:.3f}s"

def dbg_poll_times(p, now_ms_val: int, label: str):
    if not DEBUG_AUTOPILOT:
        return
    print(
        "[autopilot][dbg]",
        label,
        f"id={p.id}",
        f"symbol={p.symbol}",
        f"status={p.status}",
        f"now={now_ms_val} ({ms_to_iso(now_ms_val)})",
        f"dep_open={p.deposit_open_ms} ({ms_to_iso(p.deposit_open_ms)}, Δ{rel_secs(p.deposit_open_ms, now_ms_val)})",
        f"dep_close={p.deposit_close_ms} ({ms_to_iso(p.deposit_close_ms)}, Δ{rel_secs(p.deposit_close_ms, now_ms_val)})",
        f"tar_open={p.target_open_ms} ({ms_to_iso(p.target_open_ms)}, Δ{rel_secs(p.target_open_ms, now_ms_val)})",
        f"tar_close={p.target_close_ms} ({ms_to_iso(p.target_close_ms)}, Δ{rel_secs(p.target_close_ms, now_ms_val)})",
        flush=True,
    )
    
def now_ms() -> int:
    return int(time.time() * 1000)

def jdumps(o) -> str:
    return json.dumps(o, separators=(",", ":"))

def _log(msg: str):
    print(f"[autopilot] {msg}", flush=True)

def is_deposit_close_due(p: Poll, t_ms: int) -> bool:
    return (p.deposit_close_ms or 0) <= (t_ms - CLOCK_SKEW_MS)

def is_settlement_due(p: Poll, t_ms: int) -> bool:
    return (p.target_close_ms or 0) <= (t_ms - SETTLEMENT_GRACE_MS)

# =========================
# Phases
# =========================
def close_due_deposits(db: Session) -> int:
    t = now_ms()
    candidates = db.query(Poll).filter(Poll.status == PollStatus.OPEN).all()
    changed = 0
    for p in candidates:
        dbg_poll_times(p, t, "check-deposit-close")
        if is_deposit_close_due(p, t):
            print(f"[autopilot] closing deposits for poll {p.id} (due)", flush=True)
            take_open_snapshot_if_needed(p)
            p.status = PollStatus.DEPOSIT_CLOSED
            db.add(LiveEvent(kind="TEXT", message=f"Poll {p.id}: deposits closed",
                             ctx_json=jdumps({"poll_id": p.id, "deposit_close_ms": p.deposit_close_ms, "now_ms": t})))
            changed += 1
        else:
            print(f"[autopilot] keep OPEN poll {p.id} (not due)", flush=True)
    if changed:
        db.commit()
    return changed



def mark_due_for_settlement(db: Session) -> int:
    t = now_ms()
    candidates = db.query(Poll).filter(
        Poll.status.in_([PollStatus.DEPOSIT_CLOSED, PollStatus.SETTLING]),
        Poll.winner_side.is_(None),
    ).all()
    changed = 0
    for p in candidates:
        dbg_poll_times(p, t, "check-settlement-mark")
        if p.status != PollStatus.SETTLING and is_settlement_due(p, t):
            print(f"[autopilot] marking poll {p.id} SETTLING (due)", flush=True)
            p.status = PollStatus.SETTLING
            db.add(LiveEvent(kind="TEXT", message=f"Poll {p.id}: moving to SETTLING",
                             ctx_json=jdumps({"poll_id": p.id, "target_close_ms": p.target_close_ms, "now_ms": t})))
            changed += 1
        else:
            print(f"[autopilot] keep {p.status} poll {p.id} (not due)", flush=True)
    if changed:
        db.commit()
    return changed



def _settle_single_poll(db: Session, p: Poll) -> None:
    """
    Fully settles ONE poll:
    - Re-checks timing guard
    - Takes close snapshot
    - Computes winners + pro-rata payouts
    - Sends on-chain (if SEND_REAL_TX=true) and writes Payout rows
    - Marks poll SETTLED
    """
    t = now_ms()
    dbg_poll_times(p, t, "before-settle")
    if not is_settlement_due(p, t):
        print(f"[autopilot] SKIP settle poll {p.id} (not due on recheck)", flush=True)
        db.add(LiveEvent(kind="TEXT", message=f"Poll {p.id}: settlement skipped (not due)",
                         ctx_json=jdumps({"poll_id": p.id, "target_close_ms": p.target_close_ms, "now_ms": t})))
        db.commit()
        return


    take_open_snapshot_if_needed(p)  
    take_close_snapshot(p)           

   
    open_px = json.loads(p.open_snapshot_json)["open_price"]
    close_px = json.loads(p.close_snapshot_json)["close_price"]
    winner = decide_result(p.template_kind, open_px, close_px)


    deposits: List[Deposit] = db.query(Deposit).filter(Deposit.poll_id == p.id).all()
    total = float(sum(float(d.amount_bnb) for d in deposits))
    fee = round(total * ((p.fee_bps or 0) / 10_000.0), 8)
    winners = [d for d in deposits if d.side == winner]
    winners_sum = float(sum(float(w.amount_bnb) for w in winners))
    winner_pool = round(max(0.0, total - fee), 8)


    if fee > 0:
        try:
            txh = send_chain_transfer(TREASURY_ADDR, fee)
        except Exception as ex:
            txh = f"error:{ex}"
        db.add(Payout(
            poll_id=p.id,
            user_address=TREASURY_ADDR,
            amount_bnb=fee,
            is_house_fee=True,
            tx_hash=txh,
        ))
        db.commit()

    if winners_sum <= 0.0:
        if total > 0:
            db.add(Payout(
                poll_id=p.id,
                user_address=TREASURY_ADDR,
                amount_bnb=round(total, 8),
                is_house_fee=True,
                tx_hash="0xno_winners",
            ))
            db.commit()
    else:
        for w in winners:
            ratio = float(w.amount_bnb) / winners_sum
            pay_amt = round(winner_pool * ratio, 8)
            if pay_amt <= 0:
                continue
            try:
                txh = send_chain_transfer(w.user_address, pay_amt)
            except Exception as ex:
                txh = f"error:{ex}"
            db.add(Payout(
                poll_id=p.id,
                user_address=w.user_address,
                amount_bnb=pay_amt,
                is_house_fee=False,
                tx_hash=txh,
            ))
            db.commit()


    p.winner_side = winner
    p.status = PollStatus.SETTLED
    p.settled_at_ms = now_ms()
    db.add(LiveEvent(
        kind="TEXT",
        message=f"Poll {p.id} settled • winner: {winner}",
        ctx_json=jdumps({
            "poll_id": p.id,
            "symbol": p.symbol,
            "open": open_px,
            "close": close_px,
            "fee_bps": p.fee_bps,
            "house_fee": fee,
            "winner_pool": winner_pool,
            "winners_sum": winners_sum,
        })
    ))
    db.commit()


def settle_queue_once(db: Session, max_polls: int = 1) -> int:
    t = now_ms()
    all_candidates = db.query(Poll).filter(
        Poll.status == PollStatus.SETTLING,
        Poll.winner_side.is_(None),
    ).order_by(Poll.id.asc()).all()


    for p in all_candidates:
        dbg_poll_times(p, t, "check-settlement-due")

    todo = [p for p in all_candidates if is_settlement_due(p, t)][:max_polls]
    for p in todo:
        print(f"[autopilot] Settling poll {p.id} ({p.symbol}) …", flush=True)
        _settle_single_poll(db, p)
        print(f"[autopilot] Settled poll {p.id}", flush=True)
    return len(todo)


# =========================
# Main loop
# =========================
async def autopilot_loop():
    _log(f"starting; tick={TICK_SECONDS}s, max_settlements_per_tick={MAX_SETTLEMENTS_PER_TICK}")
    while True:
        db = None
        try:
            db = SessionLocal()
            closed  = close_due_deposits(db)
            moved   = mark_due_for_settlement(db)
            settled = settle_queue_once(db, MAX_SETTLEMENTS_PER_TICK)

            # NEW: auto-start top-3 polls (uses scoring) when safe to do so
            autopilot_seed_rounds(db)

            if closed or moved or settled:
                _log(f"tick: closed={closed}, marked={moved}, settled={settled}")
        except Exception as e:
            _log(f"error: {e}")
        finally:
            try:
                if db:
                    db.close()
            except Exception:
                pass
        await asyncio.sleep(TICK_SECONDS)

