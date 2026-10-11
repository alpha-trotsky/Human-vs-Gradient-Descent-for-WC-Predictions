"""
Copula Execution Layer Package
"""
from copula_execution.types import (
    MarketVenue, ContractSide, ActionSide, OrderType, OrderStatus,
    DisagreementType, EventKind, OrderBookLevel, OrderBook, MarketContract,
    DisagreementFlag, OrderRequest, OrderResult, Position
)
from copula_execution.risk_manager import RiskLimits, RiskManager, FeeCalculator
from copula_execution.gateway import (
    BaseExecutionGateway, PaperExecutionGateway, KalshiExecutionGateway
)
from copula_execution.ingestion import MarketIngestion
from copula_execution.coherence_scanner import CoherenceScanner
from copula_execution.classify import classify_kalshi_event, classify_polymarket_event
from copula_execution.pricer import (
    EconomicBracketPricer, EventPricer, EventPricing, FairValue, CalibrationError,
    SportsCopulaPricer, MarketMakerQuoter
)
from copula_execution.engine import ExecutionEngine

__all__ = [
    "MarketVenue",
    "ContractSide",
    "ActionSide",
    "OrderType",
    "OrderStatus",
    "DisagreementType",
    "EventKind",
    "OrderBookLevel",
    "OrderBook",
    "MarketContract",
    "DisagreementFlag",
    "OrderRequest",
    "OrderResult",
    "Position",
    "RiskLimits",
    "RiskManager",
    "FeeCalculator",
    "BaseExecutionGateway",
    "PaperExecutionGateway",
    "KalshiExecutionGateway",
    "MarketIngestion",
    "CoherenceScanner",
    "classify_kalshi_event",
    "classify_polymarket_event",
    "EconomicBracketPricer",
    "EventPricer",
    "EventPricing",
    "FairValue",
    "CalibrationError",
    "SportsCopulaPricer",
    "MarketMakerQuoter",
    "ExecutionEngine"
]
