"""
core/database.py
SQLite trade log using SQLAlchemy.
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "trades"

    trade_id        = Column(String, primary_key=True)
    entry_time      = Column(DateTime)
    exit_time       = Column(DateTime, nullable=True)
    symbol          = Column(String)
    option_type     = Column(String)   # CE | PE
    strike_price    = Column(Float)
    entry_price     = Column(Float)
    exit_price      = Column(Float, nullable=True)
    sl_price        = Column(Float)
    target_price    = Column(Float)
    lots            = Column(Integer)
    quantity        = Column(Integer)
    pnl             = Column(Float, nullable=True)
    pnl_pct         = Column(Float, nullable=True)
    won             = Column(Boolean, nullable=True)
    exit_reason     = Column(String, nullable=True)
    strategy_label  = Column(String)
    signal_score    = Column(Float)
    active_signals  = Column(Text)     # JSON list
    regime          = Column(String)
    mode            = Column(String)
    created_at      = Column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, db_path: str = "logs/trades.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)
        logger.info(f"Database initialized: {db_path}")

    def save_trade(self, trade_dict: dict):
        with self.SessionLocal() as session:
            record = TradeRecord(**trade_dict)
            session.merge(record)
            session.commit()

    def update_trade(self, trade_id: str, **kwargs):
        with self.SessionLocal() as session:
            record = session.get(TradeRecord, trade_id)
            if record:
                for k, v in kwargs.items():
                    setattr(record, k, v)
                session.commit()

    def get_today_trades(self) -> list:
        from sqlalchemy import func
        with self.SessionLocal() as session:
            today = datetime.utcnow().date()
            rows = session.query(TradeRecord).filter(
                func.date(TradeRecord.entry_time) == today
            ).all()
            return rows

    def get_today_pnl(self) -> float:
        rows = self.get_today_trades()
        return sum(r.pnl or 0 for r in rows)

    def get_all_trades(self) -> list:
        with self.SessionLocal() as session:
            return session.query(TradeRecord).order_by(TradeRecord.entry_time.desc()).all()
