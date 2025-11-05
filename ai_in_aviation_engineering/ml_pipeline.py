"""Training and prediction helpers for the demo machine-learning models."""

from __future__ import annotations

import csv
import dataclasses
import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


class _JoblibProtocol(Protocol):
    """Minimal protocol implemented by :mod:`joblib` and our fall-back."""

    def dump(self, value: Any, path: str | Path) -> Any:  # pragma: no cover - protocol
        ...

    def load(self, path: str | Path) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class _Backend:
    """Container bundling together the modelling primitives we rely on."""

    train_test_split: Callable[..., tuple]
    RandomForestRegressor: type
    RandomForestClassifier: type
    mean_absolute_error: Callable[[Iterable[Any], Iterable[Any]], float]
    accuracy_score: Callable[[Iterable[Any], Iterable[Any]], float]
    confusion_matrix: Callable[[Iterable[Any], Iterable[Any]], Any]
    joblib: _JoblibProtocol


def _load_joblib() -> _JoblibProtocol:
    try:
        return importlib.import_module("joblib")  # type: ignore[return-value]
    except Exception:  # pragma: no cover - exercised indirectly in tests
        import pickle

        class _JoblibFallback:
            @staticmethod
            def dump(value: Any, path: str | Path) -> None:
                with open(path, "wb") as buffer:
                    pickle.dump(value, buffer)

            @staticmethod
            def load(path: str | Path) -> Any:
                with open(path, "rb") as buffer:
                    return pickle.load(buffer)

        return _JoblibFallback()


def _fallback_metrics() -> tuple[
    Callable[[Iterable[Any], Iterable[Any]], float],
    Callable[[Iterable[Any], Iterable[Any]], float],
    Callable[[Iterable[Any], Iterable[Any]], Any],
]:
    def mean_absolute_error(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
        total = 0.0
        count = 0
        for truth, pred in zip(y_true, y_pred):
            total += abs(float(truth) - float(pred))
            count += 1
        return total / count if count else 0.0

    def accuracy_score(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
        matches = 0
        count = 0
        for truth, pred in zip(y_true, y_pred):
            if truth == pred:
                matches += 1
            count += 1
        return matches / count if count else 0.0

    def confusion_matrix(y_true: Iterable[Any], y_pred: Iterable[Any]) -> Any:
        labels = sorted({*y_true, *y_pred})
        index_map: MutableMapping[Any, int] = {label: idx for idx, label in enumerate(labels)}
        size = len(labels)
        matrix = [[0 for _ in range(size)] for _ in range(size)]
        for truth, pred in zip(y_true, y_pred):
            matrix[index_map[truth]][index_map[pred]] += 1
        return matrix

    return mean_absolute_error, accuracy_score, confusion_matrix


class _MeanRegressor:
    """Fallback regressor predicting the mean of the training labels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - API compat
        self._mean = 0.0

    def fit(self, _X: Any, y: Iterable[float]) -> "_MeanRegressor":
        values = list(float(value) for value in y)
        self._mean = sum(values) / len(values) if values else 0.0
        return self

    def predict(self, X: Any) -> Any:
        length = len(X)
        return [self._mean] * length


class _ModeClassifier:
    """Fallback classifier predicting the most frequent training label."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - API compat
        self._mode: Any = 0

    def fit(self, _X: Any, y: Iterable[Any]) -> "_ModeClassifier":
        counts: MutableMapping[Any, int] = {}
        for value in y:
            counts[value] = counts.get(value, 0) + 1
        if counts:
            self._mode = max(counts.items(), key=lambda item: item[1])[0]
        return self

    def predict(self, X: Any) -> Any:
        length = len(X)
        return [self._mode] * length


def _fallback_random_forests() -> tuple[type, type]:
    return _MeanRegressor, _ModeClassifier


def _fallback_train_test_split() -> Callable[..., tuple]:
    def split(
        X: Sequence[Any],
        y: Sequence[Any],
        test_size: float = 0.25,
        random_state: int | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        indices = list(range(len(X)))
        rng = random.Random(random_state)
        rng.shuffle(indices)
        pivot = int(len(indices) * (1 - test_size))
        train_idx = indices[:pivot]
        test_idx = indices[pivot:]

        def _subset(container: Sequence[Any], selected: Sequence[int]) -> list[Any]:
            return [container[idx] for idx in selected]

        return (
            _subset(X, train_idx),
            _subset(X, test_idx),
            _subset(y, train_idx),
            _subset(y, test_idx),
        )

    return split


def _to_matrix(rows: Sequence[Sequence[float]]) -> Any:
    try:
        import numpy as np  # type: ignore

        return np.array(rows, dtype=float)
    except Exception:  # pragma: no cover - numpy optional
        return [list(row) for row in rows]


def _to_vector(values: Sequence[Any], *, dtype: type = float) -> Any:
    try:
        import numpy as np  # type: ignore

        return np.array(values, dtype=dtype)
    except Exception:  # pragma: no cover - numpy optional
        if dtype is float:
            return [float(value) for value in values]
        if dtype is int:
            return [int(value) for value in values]
        return list(values)


def _resolve_backend() -> _Backend:
    joblib = _load_joblib()

    try:  # pragma: no cover - exercised when scikit-learn is installed locally
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
        from sklearn.model_selection import train_test_split

        metrics = (mean_absolute_error, accuracy_score, confusion_matrix)
        regressors = (RandomForestRegressor, RandomForestClassifier)
        split = train_test_split
    except Exception:  # pragma: no cover - fallback used in CI
        mae, acc, cm = _fallback_metrics()
        rf_regressor, rf_classifier = _fallback_random_forests()
        split = _fallback_train_test_split()
        metrics = (mae, acc, cm)
        regressors = (rf_regressor, rf_classifier)

    return _Backend(
        train_test_split=split,
        RandomForestRegressor=regressors[0],
        RandomForestClassifier=regressors[1],
        mean_absolute_error=metrics[0],
        accuracy_score=metrics[1],
        confusion_matrix=metrics[2],
        joblib=joblib,
    )


_BACKEND = _resolve_backend()

_PRICE_FEATURES = (
    "days_before_departure",
    "seat_load_factor",
    "departure_hour",
    "route_code",
)
_MAINTENANCE_FEATURES = (
    "hours_since_last_check",
    "cycles_since_last_check",
    "avg_daily_flight_hours",
)


def _coerce_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_data(filepath: str | Path) -> list[dict[str, Any]]:
    """Return a cleaned list of records containing the model training data."""

    records: list[dict[str, Any]] = []
    with open(filepath, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            coerced = {key: _coerce_value(str(value)) for key, value in row.items()}
            if any(value is None for value in coerced.values()):
                continue
            records.append(coerced)

    print(f"Data loaded from {filepath}, {len(records)} rows.")
    return records


def _as_records(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict("records")  # type: ignore[call-arg]
        except TypeError:
            pass
        else:
            if isinstance(records, list):
                return [dict(record) for record in records]
    if isinstance(data, Sequence):
        result: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise TypeError("Expected a sequence of mapping objects.")
            result.append(dict(item))
        return result
    raise TypeError("Unsupported training data format")


def _train_price_estimator(data: Any) -> tuple[Any, tuple[str, ...]]:
    records = _as_records(data)
    if not records:
        raise ValueError("Training data is empty")

    categories = tuple(sorted({record["route"] for record in records}))
    route_lookup = {route: idx for idx, route in enumerate(categories)}

    features = _to_matrix(
        [
            [
                float(record["days_before_departure"]),
                float(record["seat_load_factor"]),
                float(record["departure_hour"]),
                float(route_lookup[record["route"]]),
            ]
            for record in records
        ]
    )
    target = _to_vector(
        [float(record["final_price_eur"]) for record in records], dtype=float
    )

    X_train, X_test, y_train, y_test = _BACKEND.train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = _BACKEND.RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = _BACKEND.mean_absolute_error(y_test, predictions)
    print(f"Price model trained - MAE: {mae:.2f}")

    return model, categories


@dataclass
class PriceModelArtifacts:
    model: Any
    route_categories: tuple[str, ...]
    _route_lookup: dict[str, int] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._route_lookup = {route: idx for idx, route in enumerate(self.route_categories)}

    def save(self, directory: Path = MODELS_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _BACKEND.joblib.dump(self.model, directory / "price_model.pkl")
        _BACKEND.joblib.dump({"route_categories": list(self.route_categories)}, directory / "price_meta.pkl")

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "PriceModelArtifacts":
        model = _BACKEND.joblib.load(directory / "price_model.pkl")
        try:
            meta = _BACKEND.joblib.load(directory / "price_meta.pkl")
        except Exception:
            meta = {}
        categories = tuple(meta.get("route_categories", ()))
        return cls(model=model, route_categories=categories)

    def encode_route(self, route: str) -> int:
        return self._route_lookup.get(route, len(self.route_categories))

    def predict(
        self,
        *,
        days_before_departure: float,
        seat_load_factor: float,
        departure_hour: int,
        route: str,
    ) -> float:
        features = _to_matrix(
            [
                [
                    float(days_before_departure),
                    float(seat_load_factor),
                    float(departure_hour),
                    float(self.encode_route(route)),
                ]
            ]
        )
        prediction = self.model.predict(features)
        first = prediction[0]
        return float(first)


def train_price_model(
    data: Any,
    *,
    persist: bool = True,
    models_dir: Path = MODELS_DIR,
) -> PriceModelArtifacts:
    model, categories = _train_price_estimator(data)
    artefacts = PriceModelArtifacts(model=model, route_categories=categories)
    if persist:
        artefacts.save(models_dir)
        print(
            "Saved price model to",
            models_dir / "price_model.pkl",
            "and metadata to",
            models_dir / "price_meta.pkl",
        )
    return artefacts


def _train_maintenance_estimator(data: Any) -> Any:
    records = _as_records(data)
    if not records:
        raise ValueError("Training data is empty")

    features = _to_matrix(
        [
            [
                float(record["hours_since_last_check"]),
                float(record["cycles_since_last_check"]),
                float(record["avg_daily_flight_hours"]),
            ]
            for record in records
        ]
    )
    target = _to_vector([record["aog_flag"] for record in records], dtype=int)

    X_train, X_test, y_train, y_test = _BACKEND.train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = _BACKEND.RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = _BACKEND.accuracy_score(y_test, predictions)
    matrix = _BACKEND.confusion_matrix(y_test, predictions)

    print(f"Maintenance model trained - Accuracy: {accuracy:.2f}")
    print("Confusion matrix:")
    print(matrix)

    return model


@dataclass
class MaintenanceModelArtifacts:
    model: Any

    def save(self, directory: Path = MODELS_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _BACKEND.joblib.dump(self.model, directory / "maintenance_model.pkl")

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "MaintenanceModelArtifacts":
        model = _BACKEND.joblib.load(directory / "maintenance_model.pkl")
        return cls(model=model)

    def predict(
        self,
        *,
        hours_since_last_check: float,
        cycles_since_last_check: float,
        avg_daily_flight_hours: float,
    ) -> int:
        features = _to_matrix(
            [
                [
                    float(hours_since_last_check),
                    float(cycles_since_last_check),
                    float(avg_daily_flight_hours),
                ]
            ]
        )
        prediction = self.model.predict(features)
        first = prediction[0]
        return int(first)


def train_maintenance_model(
    data: Any,
    *,
    persist: bool = True,
    models_dir: Path = MODELS_DIR,
) -> MaintenanceModelArtifacts:
    model = _train_maintenance_estimator(data)
    artefacts = MaintenanceModelArtifacts(model=model)
    if persist:
        artefacts.save(models_dir)
        print("Saved maintenance model to", models_dir / "maintenance_model.pkl")
    return artefacts


def predict_price(
    artefacts: PriceModelArtifacts,
    *,
    days_before_departure: float,
    load_factor: float,
    departure_hour: int,
    route: str,
) -> float:
    return artefacts.predict(
        days_before_departure=days_before_departure,
        seat_load_factor=load_factor,
        departure_hour=departure_hour,
        route=route,
    )


def predict_maintenance(
    artefacts: MaintenanceModelArtifacts,
    *,
    hours_since_last_check: float,
    cycles_since_last_check: float,
    avg_daily_flight_hours: float,
) -> int:
    return artefacts.predict(
        hours_since_last_check=hours_since_last_check,
        cycles_since_last_check=cycles_since_last_check,
        avg_daily_flight_hours=avg_daily_flight_hours,
    )


def load_price_artifacts(directory: Path = MODELS_DIR) -> PriceModelArtifacts:
    return PriceModelArtifacts.load(directory)


def load_maintenance_artifacts(directory: Path = MODELS_DIR) -> MaintenanceModelArtifacts:
    return MaintenanceModelArtifacts.load(directory)


if __name__ == "__main__":  # pragma: no cover - exercised manually
    prices = load_data(DATA_DIR / "prices.csv")
    maintenance = load_data(DATA_DIR / "maintenance.csv")

    price_artifacts = train_price_model(prices)
    maintenance_artifacts = train_maintenance_model(maintenance)

    predicted_price = predict_price(
        price_artifacts,
        days_before_departure=4,
        load_factor=0.82,
        departure_hour=14,
        route="RIX-ORY",
    )
    print(f"Price prediction: {predicted_price:.2f} EUR")

    predicted_maintenance = predict_maintenance(
        maintenance_artifacts,
        hours_since_last_check=350,
        cycles_since_last_check=210,
        avg_daily_flight_hours=5.0,
    )
    print(f"Prediction maintenance (AOG flag): {predicted_maintenance}")
