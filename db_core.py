import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


DB_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("DB_URL")
    or "sqlite:///./aster_predict.db"
)

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_schema():
    
    with engine.begin() as conn:
        dups = conn.exec_driver_sql("""
            SELECT tx_hash, COUNT(*) AS cnt
            FROM deposits
            WHERE tx_hash IS NOT NULL
            GROUP BY tx_hash
            HAVING cnt > 1
        """).fetchall()
        if dups:
            print("[db] WARNING: duplicate tx_hash rows; fix before enforcing unique:", dups)

  
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_deposits_txhash_nonnull
            ON deposits (tx_hash)
            WHERE tx_hash IS NOT NULL
        """)