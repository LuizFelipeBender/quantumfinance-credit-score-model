"""QuantumFinance Credit Score API.

This module exposes the trained credit score model through a FastAPI
application. The API includes simple token based authentication and
request throttling to illustrate how partners may safely consume the
model.
"""

import os
from typing import Dict

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_TOKEN = os.getenv("API_TOKEN", "secret-token")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")


def _load_model(path: str):
    """Load the persisted sklearn pipeline."""
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")
    return joblib.load(path)


model = _load_model(MODEL_PATH)

limiter = Limiter(key_func=get_remote_address)
auth_scheme = HTTPBearer()

app = FastAPI(
    title="QuantumFinance Credit Score API",
    description="Endpoint para previsões de score de crédito.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Simple token-based auth using the Authorization header."""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
@limiter.limit("10/minute")
def health(request: Request) -> Dict[str, str]:
    """Healthcheck endpoint."""
    return {"status": "ok"}


@app.post("/predict", dependencies=[Depends(verify_token)])
@limiter.limit("5/minute")
def predict(request: Request, payload: Dict) -> Dict[str, str]:
    """Run model inference for a single record.

    The request body must contain all features expected by the model.
    """

    df = pd.DataFrame([payload])
    try:
        pred = model.predict(df)
    except Exception as exc:  # pragma: no cover - for unexpected errors
        raise HTTPException(status_code=400, detail=str(exc))
    return {"prediction": str(pred[0])}


__all__ = ["app"]

