from datetime import datetime
from enum import Enum
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    Enum as SAEnum, ForeignKey, Boolean, Index, text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ---------- Enums ----------
class PollStatus(str, Enum):
    OPEN = "OPEN"
    DEPOSIT_CLOSED = "DEPOSIT_CLOSED"
    SETTLING = "SETTLING"
    SETTLED = "SETTLED"

class RoundStatus(str, Enum):
    LIVE = "LIVE"
    COMPLETE = "COMPLETE"

# ---------- Models ----------
class Round(Base):
    __tablename__ = "rounds"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(SAEnum(RoundStatus), default=RoundStatus.LIVE, nullable=False)
    note = Column(String, default="auto")

class Poll(Base):
    __tablename__ = "polls"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=True)
    round = relationship("Round")

    symbol = Column(String, nullable=False, index=True)
    status = Column(SAEnum(PollStatus), default=PollStatus.OPEN, nullable=False)

    # template
    template_kind = Column(String, nullable=False)   
    template_label = Column(String, nullable=False)  

    # timing
    deposit_open_ms = Column(Integer, nullable=False)
    deposit_close_ms = Column(Integer, nullable=False)
    target_open_ms = Column(Integer, nullable=False)
    target_close_ms = Column(Integer, nullable=False)

    # economics
    fee_bps = Column(Integer, default=200, nullable=False)  

    # pricing context
    ref_price = Column(Float, nullable=True)

    # snapshots (json)
    open_snapshot_json = Column(Text, nullable=True)
    close_snapshot_json = Column(Text, nullable=True)

    # outcome
    winner_side = Column(String, nullable=True)  
    settled_at_ms = Column(Integer, nullable=True)

    # cached pot for UI (kept in sync on deposit)
    pot_bnb = Column(Float, default=0.0, nullable=False)
    participants = Column(Integer, default=0, nullable=False)

    # friendly question cache (optional; regenerated at serialize time if empty)
    question = Column(Text, nullable=True)

class Deposit(Base):
    __tablename__ = "deposits"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    poll_id = Column(Integer, ForeignKey("polls.id"), nullable=False, index=True)
    user_address = Column(String, nullable=False)
    side = Column(String, nullable=False)
    amount_bnb = Column(Float, nullable=False)
    tx_hash = Column(String, nullable=True, index=True)

    # add this table arg (SQLite partial unique index)
    __table_args__ = (
        Index(
            "ux_deposits_txhash_nonnull",
            "tx_hash",
            unique=True,
            sqlite_where=text("tx_hash IS NOT NULL"),
        ),
    )

class Payout(Base):
    __tablename__ = "payouts"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    poll_id = Column(Integer, ForeignKey("polls.id"), nullable=False, index=True)
    user_address = Column(String, nullable=False)
    amount_bnb = Column(Float, nullable=False)
    is_house_fee = Column(Boolean, default=False, nullable=False)
    tx_hash = Column(String, nullable=True)

class LiveEvent(Base):
    __tablename__ = "live_events"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    kind = Column(String, nullable=False)  
    message = Column(Text, nullable=False)
    ctx_json = Column(Text, nullable=True)

# ---------- Indexes ----------


Index("ix_polls_status", Poll.status)
Index("ix_polls_kind_label", Poll.template_kind, Poll.template_label)
Index("ix_polls_round", Poll.round_id)
Index("ix_polls_deposit_close", Poll.deposit_close_ms)
Index("ix_polls_target_close", Poll.target_close_ms)
Index("ix_deposits_user", Deposit.user_address)
Index("ix_deposits_poll", Deposit.poll_id)
