# settlement_router.py
import os, json, time
from typing import Literal, Optional, Tuple, List, Dict, Any

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from db_core import get_db
from models import Poll, Deposit, Payout, LiveEvent, PollStatus

from aster_price import get_live_price


from decimal import Decimal, getcontext
from web3 import Web3
from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
from web3.types import TxParams


from pathlib import Path
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = find_dotenv(filename=".env", usecwd=True) or None
    load_dotenv(env_path, override=False)
except Exception:
    pass

# ========================
# ENV / CONFIG
# ========================
SEND_REAL_TX = True
RPC = os.getenv("BSC_RPC_URL")
CHAIN_ID = int(os.getenv("CHAIN_ID", "56")) 
TREASURY_ADDR = os.getenv("TREASURY_ADDRESS") or "HOUSE_TREASURY"
TREASURY_PK = os.getenv("TREASURY_PK") or os.getenv("TREASURY_PRIVATE_KEY")
CONFIRMATIONS = int(os.getenv("CONFIRMATIONS", "0"))
ASTER_BASE = os.getenv("ASTER_BASE", "https://fapi.asterdex.com")

Side = Literal["GREEN", "RED", "UP", "DOWN"]


# ========================
# Small utils
# ========================
def now_ms() -> int:
    return int(time.time() * 1000)

def jdumps(o: Any) -> str:
    return json.dumps(o, separators=(",", ":"))

def decide_result(kind: str, open_price: float, close_price: float) -> Side:
    if kind == "CANDLE_COLOR":
        return "GREEN" if close_price >= open_price else "RED"
    return "UP" if close_price >= open_price else "DOWN"


# ========================
# Snapshot helpers
# ========================
def take_open_snapshot_if_needed(p: Poll):
    if not p.open_snapshot_json:
        p.open_snapshot_json = jdumps({
            "open_time_ms": p.target_open_ms,
            "open_price": float(get_live_price(p.symbol)),
        })
        if p.ref_price is None:
            p.ref_price = json.loads(p.open_snapshot_json)["open_price"]

def take_close_snapshot(p: Poll):
    p.close_snapshot_json = jdumps({
        "close_time_ms": now_ms(),
        "close_price": float(get_live_price(p.symbol)),
    })


# ========================
# Web3 sender (BNB)
# ========================
def _get_w3() -> Web3:
    if not RPC:
        raise RuntimeError("BSC_RPC_URL/BSC_RPC not set")
    w3 = Web3(Web3.HTTPProvider(RPC))
    try:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        pass
    if not w3.is_connected():
        raise RuntimeError(f"Could not connect to RPC: {RPC}")
    return w3

def _mock_tx(to_addr: str) -> str:
    return "0xmock_" + (to_addr[-8:] if to_addr else "house")

def send_chain_transfer(to_addr: str, amount_bnb: float) -> str:
    print(f"Sending {amount_bnb} BNB to {to_addr} (real: {SEND_REAL_TX})")
    if amount_bnb <= 0:
        raise RuntimeError("amount_bnb must be > 0")

    if not SEND_REAL_TX:
        return _mock_tx(to_addr)

    if not TREASURY_ADDR or not TREASURY_PK:
        raise RuntimeError("SEND_REAL_TX=true but TREASURY_ADDRESS / TREASURY_PK not set")

    w3 = _get_w3()
    from_cs = Web3.to_checksum_address(TREASURY_ADDR)
    to_cs   = Web3.to_checksum_address(to_addr)

    acct = w3.eth.account.from_key(TREASURY_PK)
    if acct.address.lower() != from_cs.lower():
        raise RuntimeError("TREASURY_PK does not match TREASURY_ADDRESS")

    value_wei = int(Decimal(str(amount_bnb)) * Decimal(10**18))
    nonce     = w3.eth.get_transaction_count(from_cs)

    tx: TxParams = {
        "chainId": CHAIN_ID,
        "nonce": nonce,
        "to": to_cs,
        "value": value_wei,
        "gas": 21000,
    }

    def _set_gas_fields() -> int:
        try:
            latest = w3.eth.get_block("latest")
            base = latest.get("baseFeePerGas")
            if isinstance(base, int) and base > 0:
                priority = min(Web3.to_wei(1, "gwei"), base)  
                tx["maxPriorityFeePerGas"] = priority
                tx["maxFeePerGas"] = base + priority
                return (base + priority) * tx["gas"]
        except Exception:
            pass
    
        gp = w3.eth.gas_price or Web3.to_wei(1, "gwei")
        if gp < Web3.to_wei(1, "gwei"):
            gp = Web3.to_wei(1, "gwei")
        tx["gasPrice"] = gp
        return gp * tx["gas"]

    est_fee = _set_gas_fields()

    bal = w3.eth.get_balance(from_cs)
    if bal < value_wei + est_fee:
        have = Decimal(bal) / Decimal(1e18)
        need = Decimal(value_wei + est_fee) / Decimal(1e18)
        raise RuntimeError(f"Insufficient balance: need ~{need:.8f} BNB, have {have:.8f} BNB")

    signed = w3.eth.account.sign_transaction(tx, private_key=TREASURY_PK)
    
    txh = w3.eth.send_raw_transaction(signed.raw_transaction).hex()

    if CONFIRMATIONS > 0:
        w3.eth.wait_for_transaction_receipt(txh, timeout=180)

    return txh


# ========================
# API payloads
# ========================
class SettleResponse(BaseModel):
    ok: bool
    poll_id: int
    winner: Side
    total_pot_bnb: float
    house_fee_bnb: float
    winner_pool_bnb: float
    payouts_created: int


# ========================
# Router
# ========================
router = APIRouter(prefix="/admin", tags=["admin"])


def _compute_and_payout(db: Session, p: Poll):
    take_open_snapshot_if_needed(p)
    take_close_snapshot(p)

    open_px = json.loads(p.open_snapshot_json)["open_price"]
    close_px = json.loads(p.close_snapshot_json)["close_price"]
    winner = decide_result(p.template_kind, open_px, close_px)

    rows = db.query(Deposit).filter(Deposit.poll_id == p.id).all()
    total = float(sum(float(d.amount_bnb) for d in rows))
    fee_bps = float(p.fee_bps or 0)
    fee = round(total * (fee_bps / 10_000.0), 8)

    winners = [d for d in rows if d.side == winner]
    winners_sum = float(sum(float(w.amount_bnb) for w in winners))
    winner_pool = round(max(0.0, total - fee), 8)

    if fee > 0:
        try:
            txh_house = send_chain_transfer(TREASURY_ADDR, fee)
        except Exception as ex:
            txh_house = f"error:{ex}"
        db.add(Payout(
            poll_id=p.id,
            user_address=TREASURY_ADDR, 
            amount_bnb=fee,
            is_house_fee=True,
            tx_hash=txh_house,
        ))

    if winners_sum <= 0.0:
        if total > 0:
            db.add(Payout(
                poll_id=p.id,
                user_address=TREASURY_ADDR,
                amount_bnb=round(total, 8),
                is_house_fee=True,
                tx_hash="0xno_winners",
            ))
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

    p.winner_side = winner
    p.status = PollStatus.SETTLED
    p.settled_at_ms = now_ms()
    db.commit()

    db.add(LiveEvent(
        kind="TEXT",
        message=f"Poll {p.id} settled • winner: {winner} • pot: {round(total,4)} BNB",
        ctx_json=jdumps({
            "poll_id": p.id,
            "symbol": p.symbol,
            "open": open_px, "close": close_px,
            "fee_bps": p.fee_bps,
            "house_fee": fee,
            "winner_pool": winner_pool,
            "winners_sum": winners_sum,
            "send_real_tx": SEND_REAL_TX,
            "chain_id": CHAIN_ID,
        })
    ))
    db.commit()

    return total, fee, winner_pool, winner


@router.post("/polls/{poll_id}/settle", response_model=SettleResponse)
def settle_poll(poll_id: int, db: Session = Depends(get_db)):
    p = db.query(Poll).get(poll_id)
    print(f"send real tx: {SEND_REAL_TX}")
    if not p:
        raise HTTPException(status_code=404, detail="poll not found")

    
    if p.winner_side:
        total = db.query(func.coalesce(func.sum(Deposit.amount_bnb), 0.0)).filter(Deposit.poll_id == p.id).scalar() or 0.0
        fee = float(total) * (float(p.fee_bps or 0) / 10_000.0)
        return SettleResponse(
            ok=True,
            poll_id=p.id,
            winner=p.winner_side,  
            total_pot_bnb=round(float(total), 8),
            house_fee_bnb=round(float(fee), 8),
            winner_pool_bnb=round(float(total - fee), 8),
            payouts_created=db.query(Payout).filter(Payout.poll_id == p.id).count(),
        )


    if p.status in (PollStatus.OPEN, PollStatus.DEPOSIT_CLOSED):
        p.status = PollStatus.SETTLING
        db.commit()

    total, fee, winner_pool, winner = _compute_and_payout(db, p)

    return SettleResponse(
        ok=True,
        poll_id=p.id,
        winner=winner,  
        total_pot_bnb=round(float(total), 8),
        house_fee_bnb=round(float(fee), 8),
        winner_pool_bnb=round(float(winner_pool), 8),
        payouts_created=db.query(Payout).filter(Payout.poll_id == p.id).count(),
    )
