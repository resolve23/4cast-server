from __future__ import annotations

import os
import time
from decimal import Decimal
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv, find_dotenv
from web3 import Web3
from web3.types import TxReceipt

load_dotenv(find_dotenv(), override=False)

# ---------- config ----------
BSC_RPC_URL = os.getenv("BSC_RPC_URL")
ALT_RPCS = [
    u.strip() for u in (os.getenv("BSC_ALT_RPC_URLS") or "").split(",") if u.strip()
]
CHAIN_ID = int(os.getenv("CHAIN_ID") or 56)
CONFIRMATIONS = int(os.getenv("CONFIRMATIONS") or 1)

TREASURY_ADDRESS_ENV = os.getenv("TREASURY_ADDRESS") or ""
TREASURY_ADDRESS = (
    Web3.to_checksum_address(TREASURY_ADDRESS_ENV) if TREASURY_ADDRESS_ENV else None
)

DEFAULT_TIMEOUT_S = int(os.getenv("TX_VERIFY_TIMEOUT_S") or 45)
POLL_INTERVAL_S = float(os.getenv("TX_VERIFY_POLL_INTERVAL_S") or 1.5)

# ---------- utilities ----------
class OnChainVerifyError(ValueError):
    """Raised for user-facing verification errors (safe to bubble to API)."""


def _providers() -> List[Web3]:
    urls = [u for u in [BSC_RPC_URL] + ALT_RPCS if u]
    if not urls:
        raise RuntimeError(
            "No RPC URLs configured. Set BSC_RPC_URL (and optional BSC_ALT_RPC_URLS)."
        )
    return [
        Web3(Web3.HTTPProvider(u, request_kwargs={"timeout": 10})) for u in urls
    ]


def _try_get_tx(w3: Web3, txh: str):
    try:
        return w3.eth.get_transaction(txh)
    except Exception:
        return None


def _try_get_receipt(w3: Web3, txh: str) -> Optional[TxReceipt]:
    try:
        return w3.eth.get_transaction_receipt(txh)
    except Exception:
        return None


def _wait_for_tx_any(
    txh: str, timeout_s: int = DEFAULT_TIMEOUT_S, poll_s: float = POLL_INTERVAL_S
) -> Tuple[Web3, Optional[TxReceipt]]:
    """
    Poll all configured providers until a receipt appears or timeout.
    Returns (provider_that_saw_it, receipt_or_None).
    """
    txh = Web3.to_hex(hexstr=txh)
    start = time.time()
    w3s = _providers()
    last_w3: Optional[Web3] = None

    while time.time() - start < timeout_s:
        for w3 in w3s:
            rcpt = _try_get_receipt(w3, txh)
            if rcpt:
                return (w3, rcpt)
            tx = _try_get_tx(w3, txh)
            if tx:
                last_w3 = w3
        time.sleep(poll_s)

    return (last_w3 or w3s[0], None)


def _has_confirmations(w3: Web3, rcpt: TxReceipt, min_conf: int) -> bool:
    if rcpt.blockNumber is None:
        return False
    latest = w3.eth.block_number
    return (latest - rcpt.blockNumber) + 1 >= max(1, min_conf)


def to_wei_bnb(amount_bnb: Decimal | float | str | int) -> int:
    d = Decimal(str(amount_bnb))
    return int(d * Decimal(10**18))


def from_wei_bnb(amount_wei: int) -> Decimal:
    return Decimal(amount_wei) / Decimal(10**18)


# ---------- public verification ----------
def verify_native_deposit(
    tx_hash: str,
    expected_to: str,
    expected_amount_wei: int,
    *,
    wait_for_confirmations: bool = True,
) -> Dict[str, Any]:

    if not expected_to:
        raise OnChainVerifyError("backend misconfiguration: TREASURY_ADDRESS is empty")

    try:
        exp_to_cs = Web3.to_checksum_address(expected_to)
    except Exception:
        raise OnChainVerifyError("backend misconfiguration: invalid TREASURY_ADDRESS")

    try:
        tx_hash = Web3.to_hex(hexstr=tx_hash)
    except Exception:
        raise OnChainVerifyError("invalid tx hash format")

    w3, rcpt = _wait_for_tx_any(tx_hash)

    if rcpt is None:
        raise OnChainVerifyError(
            "on-chain verification failed: transaction not found (still propagating?)"
        )

    if rcpt.status != 1:
        raise OnChainVerifyError("on-chain verification failed: transaction reverted")

    
    if wait_for_confirmations and not _has_confirmations(w3, rcpt, CONFIRMATIONS):
        raise OnChainVerifyError(
            f"on-chain verification pending: waiting for {CONFIRMATIONS} confirmation(s)"
        )

    tx = w3.eth.get_transaction(tx_hash)
    to_addr = tx["to"] and Web3.to_checksum_address(tx["to"])
    value_wei = int(tx["value"])

    if to_addr != exp_to_cs:
        raise OnChainVerifyError(
            f"on-chain verification failed: recipient mismatch ({to_addr} != {exp_to_cs})"
        )

    if value_wei < expected_amount_wei:
        raise OnChainVerifyError(
            f"on-chain verification failed: value {value_wei} wei < expected {expected_amount_wei} wei"
        )

    confs = 0
    if rcpt.blockNumber is not None:
        confs = (w3.eth.block_number - rcpt.blockNumber) + 1

    return {
        "hash": tx_hash,
        "chain_id": w3.eth.chain_id,
        "block_number": rcpt.blockNumber,
        "to": to_addr,
        "value_wei": value_wei,
        "value_bnb": str(from_wei_bnb(value_wei)),
        "confirmations": confs,
    }


# ---------- convenience for your /deposits handler ----------
def verify_deposit_for_backend(tx_hash: str, amount_bnb: Decimal | float | str) -> Dict[str, Any]:
    if not TREASURY_ADDRESS:
        raise OnChainVerifyError("backend misconfiguration: TREASURY_ADDRESS not set")

    expected_wei = to_wei_bnb(amount_bnb)
    return verify_native_deposit(
        tx_hash=tx_hash,
        expected_to=TREASURY_ADDRESS,
        expected_amount_wei=expected_wei,
        wait_for_confirmations=True,
    )
