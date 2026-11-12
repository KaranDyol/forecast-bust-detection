"""
db.py
=====
Hybrid SQLAlchemy 2.0 persistence layer for the Forecast Bust Detection platform.

- Defaults to a zero-config local SQLite database (`forecast_bust.db`) in WAL mode.
- Switches automatically to PostgreSQL (`postgresql+psycopg://`) whenever `DATABASE_URL`
  is configured via environment variables or Streamlit secrets (e.g., Supabase or Neon).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
import os
from urllib.parse import urlparse

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from stations_catalog import INDIAN_MET_STATIONS

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = BASE_DIR / "forecast_bust.db"
_ENGINE_CACHE: Dict[str, Engine] = {}


class Base(DeclarativeBase):
    pass


class Station(Base):
    __tablename__ = "stations"

    station_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    state: Mapped[str] = mapped_column(String(80), nullable=False)
    climatic_zone: Mapped[str] = mapped_column(String(80), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_coastal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ForecastRun(Base):
    __tablename__ = "forecast_runs"
    __table_args__ = (
        UniqueConstraint("station_id", "nwp_model", "issued_date", "valid_date", name="uq_forecast_snapshot"),
        Index("idx_forecast_valid", "valid_date", "station_id", "lead_day"),
    )

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(ForeignKey("stations.station_id"), nullable=False)
    nwp_model: Mapped[str] = mapped_column(String(60), nullable=False)
    issued_date: Mapped[date] = mapped_column(Date, nullable=False)
    valid_date: Mapped[date] = mapped_column(Date, nullable=False)
    lead_day: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_temp: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_rainfall: Mapped[float] = mapped_column(Float, nullable=False)
    wind_shear: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_drop: Mapped[float] = mapped_column(Float, nullable=False)
    cape: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relative_humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_temp_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_rain_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_bust_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class VerificationOutcome(Base):
    __tablename__ = "verification_outcomes"
    __table_args__ = (
        UniqueConstraint("station_id", "obs_date", "source", name="uq_verification_obs"),
        Index("idx_verification_date", "obs_date", "station_id"),
    )

    verification_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(ForeignKey("stations.station_id"), nullable=False)
    obs_date: Mapped[date] = mapped_column(Date, nullable=False)
    actual_temp: Mapped[float] = mapped_column(Float, nullable=False)
    actual_rainfall: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(50), default="open_meteo_day0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = (Index("idx_logs_requested_at", "requested_at"),)

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    station_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lead_time_hours: Mapped[int] = mapped_column(Integer, default=72, nullable=False)
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    risk_prob_pct: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False)
    shap_factors: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="api", nullable=False)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_version: Mapped[str] = mapped_column(String(80), primary_key=True)
    champion_name: Mapped[str] = mapped_column(String(80), nullable=False)
    training_source: Mapped[str] = mapped_column(String(160), nullable=False)
    train_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    bust_rate: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    global_drivers: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


def _normalize_database_url(raw_url: str) -> str:
    url = raw_url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def resolve_database_url(database_url: Optional[str] = None) -> str:
    """Resolve DATABASE_URL from arg, env, Streamlit secrets, or local SQLite fallback."""
    if database_url:
        return _normalize_database_url(database_url)

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return _normalize_database_url(env_url)

    try:
        import streamlit as st

        secret_url = st.secrets.get("DATABASE_URL")
        if secret_url:
            return _normalize_database_url(str(secret_url))
    except Exception:
        pass

    return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def get_engine(database_url: Optional[str] = None) -> Engine:
    """Return a cached SQLAlchemy engine configured for SQLite or PostgreSQL."""
    url = resolve_database_url(database_url)
    if url in _ENGINE_CACHE:
        return _ENGINE_CACHE[url]

    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _conn_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")
            cursor.close()
    else:
        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            future=True,
        )

    _ENGINE_CACHE[url] = engine
    return engine


@contextmanager
def session_scope(engine: Optional[Engine] = None) -> Iterator[Session]:
    """Transactional session context manager."""
    eng = engine or get_engine()
    factory = sessionmaker(bind=eng, expire_on_commit=False, future=True)
    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _dialect_upsert(
    session: Session,
    model_cls: Any,
    rows: Sequence[Dict[str, Any]],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
    chunk_size: int = 50,
) -> int:
    """Execute idempotent upsert batches across SQLite and PostgreSQL."""
    if not rows:
        return 0

    deduped_map = {tuple(row[col] for col in conflict_cols): row for row in rows}
    deduped_rows = list(deduped_map.values())

    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
    insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert

    total = 0
    for idx in range(0, len(deduped_rows), chunk_size):
        chunk = list(deduped_rows[idx : idx + chunk_size])
        stmt = insert_fn(model_cls).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=list(conflict_cols),
            set_={col: stmt.excluded[col] for col in update_cols},
        )
        session.execute(stmt)
        total += len(chunk)
    return total


def init_db(engine: Optional[Engine] = None) -> Engine:
    """Create tables and the cross-dialect paired verification view."""
    eng = engine or get_engine()
    Base.metadata.create_all(eng)

    view_body = """
        SELECT
            f.run_id,
            s.station_id,
            s.name AS station_name,
            s.state,
            s.climatic_zone,
            s.is_coastal,
            f.nwp_model,
            f.issued_date,
            f.valid_date,
            f.lead_day,
            f.predicted_temp,
            v.actual_temp,
            f.predicted_rainfall,
            v.actual_rainfall,
            f.wind_shear,
            f.pressure_drop,
            f.cape,
            f.relative_humidity,
            f.model_temp_spread,
            f.model_rain_spread,
            f.predicted_bust_prob,
            ABS(f.predicted_temp - v.actual_temp) AS temp_error,
            ABS(f.predicted_rainfall - v.actual_rainfall) AS rainfall_error,
            CASE
                WHEN ABS(f.predicted_temp - v.actual_temp) > 3.0
                  OR ABS(f.predicted_rainfall - v.actual_rainfall) > 15.0
                THEN 1 ELSE 0
            END AS is_bust
        FROM forecast_runs f
        JOIN verification_outcomes v
          ON f.station_id = v.station_id
         AND f.valid_date = v.obs_date
        JOIN stations s
          ON f.station_id = s.station_id
    """
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            conn.execute(text(f"CREATE OR REPLACE VIEW v_verified_forecasts AS {view_body}"))
        else:
            conn.execute(text(f"CREATE VIEW IF NOT EXISTS v_verified_forecasts AS {view_body}"))
    return eng


def seed_stations(
    elevations: Optional[Dict[str, float]] = None,
    engine: Optional[Engine] = None,
) -> Dict[str, int]:
    """Upsert all canonical Indian stations and return name -> station_id mapping."""
    eng = init_db(engine)
    elev_map = elevations or {}
    payload = [
        {
            "name": s["name"],
            "state": s["state"],
            "climatic_zone": s["zone"],
            "latitude": float(s["lat"]),
            "longitude": float(s["lon"]),
            "elevation_m": float(elev_map[s["name"]]) if s["name"] in elev_map else None,
            "is_coastal": bool(s.get("is_coastal", False)),
            "is_active": True,
        }
        for s in INDIAN_MET_STATIONS
    ]
    with session_scope(eng) as session:
        update_cols = ["state", "climatic_zone", "latitude", "longitude", "is_coastal", "is_active"]
        if elev_map:
            update_cols.append("elevation_m")
        _dialect_upsert(
            session=session,
            model_cls=Station,
            rows=payload,
            conflict_cols=["name"],
            update_cols=update_cols,
        )
        rows = session.execute(select(Station.name, Station.station_id)).all()
    return {name: sid for name, sid in rows}


def upsert_forecast_snapshots(
    rows: Sequence[Dict[str, Any]],
    nwp_model: Optional[str] = None,
    issued_date: Optional[date] = None,
    model_version: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> int:
    """Upsert forward or historical NWP forecast snapshots keyed cleanly by date and lead."""
    if not rows:
        return 0

    eng = init_db(engine)
    station_map = seed_stations(engine=eng)
    default_issue = issued_date or datetime.now(timezone.utc).date()

    prepared: List[Dict[str, Any]] = []
    for row in rows:
        sid = station_map.get(str(row.get("station_name", "")))
        if sid is None:
            continue
        valid_d = _coerce_date(row.get("valid_date") or row.get("forecast_date"))
        issue_d = _coerce_date(row["issued_date"]) if row.get("issued_date") else default_issue
        prepared.append({
            "station_id": sid,
            "nwp_model": str(row.get("nwp_model") or nwp_model or "ecmwf_ifs025"),
            "issued_date": issue_d,
            "valid_date": valid_d,
            "lead_day": int(row["lead_day"]),
            "predicted_temp": float(row["predicted_temp"]),
            "predicted_rainfall": float(row["predicted_rainfall"]),
            "wind_shear": float(row["wind_shear"]),
            "pressure_drop": float(row["pressure_drop"]),
            "cape": float(row["cape"]) if row.get("cape") is not None else None,
            "relative_humidity": (
                float(row["relative_humidity"]) if row.get("relative_humidity") is not None else None
            ),
            "model_temp_spread": (
                float(row["model_temp_spread"]) if row.get("model_temp_spread") is not None else None
            ),
            "model_rain_spread": (
                float(row["model_rain_spread"]) if row.get("model_rain_spread") is not None else None
            ),
            "predicted_bust_prob": (
                float(row["predicted_bust_prob"])
                if row.get("predicted_bust_prob") is not None
                else (float(row["risk_percentage"]) / 100.0 if row.get("risk_percentage") is not None else None)
            ),
            "model_version": row.get("model_version") or model_version,
            "created_at": datetime.now(timezone.utc),
        })

    if not prepared:
        return 0

    with session_scope(eng) as session:
        return _dialect_upsert(
            session=session,
            model_cls=ForecastRun,
            rows=prepared,
            conflict_cols=["station_id", "nwp_model", "issued_date", "valid_date"],
            update_cols=[
                "lead_day",
                "predicted_temp",
                "predicted_rainfall",
                "wind_shear",
                "pressure_drop",
                "cape",
                "relative_humidity",
                "model_temp_spread",
                "model_rain_spread",
                "predicted_bust_prob",
                "model_version",
            ],
        )


def upsert_verification_outcomes(
    rows: Sequence[Dict[str, Any]],
    source: str = "open_meteo_day0",
    engine: Optional[Engine] = None,
) -> int:
    """Upsert ground-truth daily observations keyed by station, date, and source."""
    if not rows:
        return 0

    eng = init_db(engine)
    station_map = seed_stations(engine=eng)

    prepared: List[Dict[str, Any]] = []
    for row in rows:
        sid = station_map.get(str(row.get("station_name", "")))
        if sid is None:
            continue
        obs_d = _coerce_date(row.get("obs_date") or row.get("valid_date"))
        prepared.append({
            "station_id": sid,
            "obs_date": obs_d,
            "actual_temp": float(row["actual_temp"]),
            "actual_rainfall": float(row["actual_rainfall"]),
            "source": str(row.get("source") or source),
            "created_at": datetime.now(timezone.utc),
        })

    if not prepared:
        return 0

    with session_scope(eng) as session:
        return _dialect_upsert(
            session=session,
            model_cls=VerificationOutcome,
            rows=prepared,
            conflict_cols=["station_id", "obs_date", "source"],
            update_cols=["actual_temp", "actual_rainfall"],
        )


def log_prediction(
    input_features: Dict[str, Any],
    risk_prob_pct: float,
    risk_level: str,
    shap_factors: List[Dict[str, Any]],
    model_version: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    station_name: Optional[str] = None,
    lead_time_hours: int = 72,
    source: str = "api",
    engine: Optional[Engine] = None,
) -> None:
    """Persist an inference event and its SHAP explanation for auditability."""
    eng = init_db(engine)
    with session_scope(eng) as session:
        session.add(
            PredictionLog(
                station_name=station_name,
                latitude=latitude,
                longitude=longitude,
                lead_time_hours=int(lead_time_hours),
                input_features=dict(input_features),
                risk_prob_pct=float(risk_prob_pct),
                risk_level=str(risk_level),
                shap_factors=list(shap_factors),
                model_version=str(model_version),
                source=str(source),
            )
        )


def register_model_run(
    model_version: str,
    champion_name: str,
    training_source: str,
    train_samples: int,
    bust_rate: float,
    metrics: Dict[str, Any],
    global_drivers: List[Dict[str, Any]],
    artifact_uri: str = "bust_model.pkl",
    set_active: bool = True,
    engine: Optional[Engine] = None,
) -> None:
    """Record a trained model version and promote it as active in the registry."""
    eng = init_db(engine)
    row = {
        "model_version": str(model_version),
        "champion_name": str(champion_name),
        "training_source": str(training_source),
        "train_samples": int(train_samples),
        "bust_rate": float(bust_rate),
        "roc_auc": float(metrics["roc_auc"]) if metrics.get("roc_auc") is not None else None,
        "f1_score": float(metrics["f1"]) if metrics.get("f1") is not None else None,
        "precision_score": float(metrics["precision"]) if metrics.get("precision") is not None else None,
        "recall_score": float(metrics["recall"]) if metrics.get("recall") is not None else None,
        "brier_score": float(metrics["brier_score"]) if metrics.get("brier_score") is not None else None,
        "global_drivers": list(global_drivers),
        "artifact_uri": str(artifact_uri),
        "is_active": bool(set_active),
        "trained_at": datetime.now(timezone.utc),
    }
    with session_scope(eng) as session:
        if set_active:
            session.query(ModelRegistry).update({ModelRegistry.is_active: False})
        _dialect_upsert(
            session=session,
            model_cls=ModelRegistry,
            rows=[row],
            conflict_cols=["model_version"],
            update_cols=[
                "champion_name",
                "training_source",
                "train_samples",
                "bust_rate",
                "roc_auc",
                "f1_score",
                "precision_score",
                "recall_score",
                "brier_score",
                "global_drivers",
                "artifact_uri",
                "is_active",
                "trained_at",
            ],
        )


def fetch_verified_training_records(
    nwp_model: Optional[str] = None,
    limit: Optional[int] = None,
    engine: Optional[Engine] = None,
) -> List[Dict[str, Any]]:
    """Query joined forecast-vs-observation rows from `v_verified_forecasts`."""
    eng = init_db(engine)
    sql = "SELECT * FROM v_verified_forecasts"
    params: Dict[str, Any] = {}
    if nwp_model:
        sql += " WHERE nwp_model = :nwp_model"
        params["nwp_model"] = nwp_model
    sql += " ORDER BY valid_date DESC, station_id ASC"
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    with eng.connect() as conn:
        result = conn.execute(text(sql), params)
        return [dict(r._mapping) for r in result]


def _safe_location_label(url: str) -> str:
    """Return a non-secret display label for the connected database."""
    if url.startswith("sqlite"):
        return DEFAULT_SQLITE_PATH.name if DEFAULT_SQLITE_PATH.as_posix() in url else "SQLite local file"
    parsed = urlparse(url)
    host = parsed.hostname or "postgres-host"
    db_name = parsed.path.lstrip("/") or "postgres"
    return f"{host}/{db_name}"


def get_database_summary(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Return connection status and row counts across all operational tables."""
    eng = init_db(engine)
    url = str(eng.url)
    backend = "PostgreSQL" if eng.dialect.name == "postgresql" else "SQLite"

    with session_scope(eng) as session:
        stations_n = session.scalar(select(func.count()).select_from(Station)) or 0
        forecasts_n = session.scalar(select(func.count()).select_from(ForecastRun)) or 0
        verif_n = session.scalar(select(func.count()).select_from(VerificationOutcome)) or 0
        logs_n = session.scalar(select(func.count()).select_from(PredictionLog)) or 0
        models_n = session.scalar(select(func.count()).select_from(ModelRegistry)) or 0
        paired_n = session.execute(text("SELECT COUNT(*) FROM v_verified_forecasts")).scalar() or 0

    return {
        "backend": backend,
        "location": _safe_location_label(url),
        "stations": int(stations_n),
        "forecast_runs": int(forecasts_n),
        "verification_outcomes": int(verif_n),
        "verified_pairs": int(paired_n),
        "prediction_logs": int(logs_n),
        "model_runs": int(models_n),
    }


def bootstrap_database(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Initialize schema, ensure the 18 synoptic stations exist, and return counts."""
    eng = init_db(engine)
    seed_stations(engine=eng)
    return get_database_summary(engine=eng)
