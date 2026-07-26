from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "transportation": "transportation_bundle.joblib",
    "energy": "energy_bundle.joblib",
    "governance": "governance_bundle.joblib",
    "waste": "waste_bundle.joblib",
}

MODEL_BUNDLES: dict[str, Any] = {}
MODEL_ERRORS: dict[str, str | None] = {}


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
def load_models() -> None:
    """Load every model once when the Flask application starts."""
    for module_name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name

        if not model_path.exists():
            MODEL_BUNDLES[module_name] = None
            MODEL_ERRORS[module_name] = f"File not found: models/{file_name}"
            print(
                f"[MODEL] {module_name}: FILE NOT FOUND -> {model_path}",
                flush=True,
            )
            continue

        try:
            MODEL_BUNDLES[module_name] = joblib.load(model_path)
            MODEL_ERRORS[module_name] = None
            print(
                f"[MODEL] {module_name}: LOADED SUCCESSFULLY | "
                f"type={type(MODEL_BUNDLES[module_name]).__name__}",
                flush=True,
            )
        except Exception as error:
            MODEL_BUNDLES[module_name] = None
            MODEL_ERRORS[module_name] = (
                f"{type(error).__name__}: {error}"
            )
            print(
                f"[MODEL] {module_name}: LOAD FAILED | "
                f"{type(error).__name__}: {error}",
                flush=True,
            )


def transportation_model_ready() -> bool:
    """Transportation is a two-stage hurdle model."""
    bundle = MODEL_BUNDLES.get("transportation")

    return (
        isinstance(bundle, dict)
        and bundle.get("stage1_model") is not None
        and bundle.get("stage2_model") is not None
        and bool(bundle.get("feature_columns"))
    )


def generic_model_ready(module_name: str) -> bool:
    """Check common single-model bundle formats."""
    bundle = MODEL_BUNDLES.get(module_name)

    if bundle is None:
        return False

    if hasattr(bundle, "predict"):
        return True

    if isinstance(bundle, dict):
        for key in (
            "model",
            "pipeline",
            "estimator",
            "best_model",
            "best_estimator",
            "regressor",
            "classifier",
        ):
            candidate = bundle.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return True

    return False


def model_found(module_name: str) -> bool:
    if module_name == "transportation":
        return transportation_model_ready()

    return generic_model_ready(module_name)


def get_model_status() -> dict[str, bool]:
    return {
        module_name: model_found(module_name)
        for module_name in MODEL_FILES
    }


# ---------------------------------------------------------
# Transportation preprocessing and prediction
# ---------------------------------------------------------
def _form_float(name: str, default: float) -> float:
    value = request.form.get(name, "").strip()
    return float(value) if value else float(default)


def _form_int(name: str, default: int) -> int:
    value = request.form.get(name, "").strip()
    return int(value) if value else int(default)


def prepare_transportation_input() -> pd.DataFrame:
    """
    Build one row with the exact 53 training features stored in the bundle.
    All one-hot columns begin at zero, then the selected category is set to one.
    """
    bundle = MODEL_BUNDLES.get("transportation")

    if not transportation_model_ready():
        raise RuntimeError(
            "Transportation bundle is not ready. It must contain "
            "stage1_model, stage2_model, and feature_columns."
        )

    feature_columns = list(bundle["feature_columns"])
    row: dict[str, float | int] = {
        feature: 0 for feature in feature_columns
    }

    numeric_values = {
        "air_co": _form_float("air_co", 0),
        "air_nox": _form_float("air_nox", 40),
        "air_no2": _form_float("air_no2", 30),
        "air_no": _form_float("air_no", 10),
        "air_o3": _form_float("air_o3", 40),
        "air_air_temp": _form_float("air_air_temp", 15),
        "latitude": _form_float("latitude", 51.5074),
        "longitude": _form_float("longitude", -0.1278),
        "weather_tavg": _form_float("weather_tavg", 15),
        "weather_tmin": _form_float("weather_tmin", 10),
        "weather_tmax": _form_float("weather_tmax", 20),
        "weather_prcp": _form_float("weather_prcp", 0),
        "weather_wdir": _form_float("weather_wdir", 180),
        "weather_wspd": _form_float("weather_wspd", 10),
        "weather_pres": _form_float("weather_pres", 1013),
        "bike_cnt": _form_float("bike_cnt", 1000),
        "bike_t1": _form_float("bike_t1", 15),
        "bike_t2": _form_float("bike_t2", 14),
        "bike_hum": _form_float("bike_hum", 70),
        "bike_wind_speed": _form_float("bike_wind_speed", 10),
        "bike_weather_code": _form_float("bike_weather_code", 1),
        "bike_is_holiday": _form_int("bike_is_holiday", 0),
        "year": _form_int("year", 2026),
        "month": _form_int("month", 1),
        "hour": _form_int("hour", 12),
        "day_of_week": _form_int("day_of_week", 0),
        "is_weekend": _form_int("is_weekend", 0),
    }

    for feature, value in numeric_values.items():
        if feature in row:
            row[feature] = value

    categorical_selections = {
        "site": request.form.get("site", "London Bloomsbury"),
        "code": request.form.get("code", "CLL2"),
        "site_type": request.form.get(
            "site_type",
            "Urban Background",
        ),
        "season": request.form.get("season", "Winter"),
    }

    for prefix, selected_value in categorical_selections.items():
        selected_column = f"{prefix}_{selected_value}"
        if selected_column not in row:
            raise ValueError(
                f"Unsupported {prefix} value: {selected_value}"
            )
        row[selected_column] = 1

    input_frame = pd.DataFrame([row], columns=feature_columns)

    # Keep numeric values numeric and preserve exact training order.
    input_frame = input_frame.apply(pd.to_numeric, errors="raise")
    return input_frame


def run_transportation_prediction(
    input_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Stage 1 predicts whether collision activity is expected.
    Stage 2 predicts the collision count when stage 1 passes the threshold.
    """
    bundle = MODEL_BUNDLES.get("transportation")

    if not transportation_model_ready():
        raise RuntimeError("Transportation model is not connected.")

    stage1_model = bundle["stage1_model"]
    stage2_model = bundle["stage2_model"]
    threshold = float(
        bundle.get("classification_threshold", 0.5)
    )

    if hasattr(stage1_model, "predict_proba"):
        probability = float(
            stage1_model.predict_proba(input_frame)[0][1]
        )
    else:
        stage1_prediction = float(
            stage1_model.predict(input_frame)[0]
        )
        probability = max(0.0, min(1.0, stage1_prediction))

    collision_expected = probability >= threshold

    if collision_expected:
        predicted_count = float(
            stage2_model.predict(input_frame)[0]
        )
        predicted_count = max(0.0, predicted_count)
    else:
        predicted_count = 0.0

    probability_percent = probability * 100

    if probability_percent >= 70:
        risk_level = "High"
        status = "Immediate traffic-safety attention recommended"
    elif probability_percent >= 40:
        risk_level = "Moderate"
        status = "Preventive monitoring recommended"
    else:
        risk_level = "Low"
        status = "Normal monitoring is sufficient"

    if collision_expected:
        headline = (
            f"Collision activity is possible — "
            f"approximately {predicted_count:.1f} collisions expected"
        )
    else:
        headline = "Low probability of collision activity"

    return {
        "collision_probability": probability,
        "probability_percent": probability_percent,
        "collision_expected": collision_expected,
        "predicted_collision_count": predicted_count,
        "threshold": threshold,
        "risk_level": risk_level,
        "status": status,
        "headline": headline,
        "model_name": bundle.get(
            "model_name",
            "Hurdle LightGBM",
        ),
        "unit": bundle.get("unit", "collisions"),
    }


# Load models before serving requests.
load_models()


# ---------------------------------------------------------
# Shared template data
# ---------------------------------------------------------
@app.context_processor
def inject_global_values() -> dict[str, Any]:
    return {
        "current_year": datetime.now().year,
    }


def default_profile() -> dict[str, Any]:
    return {
        "city_name": "London",
        "country": "United Kingdom",
        "population": 8_900_000,
        "districts": 32,
        "role": "City Manager",
    }


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    profile = session.get("profile", default_profile())

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name", "City Manager"),
        profile=profile,
        model_status=get_model_status(),
        results=session.get("results", {}),
        waste=session.get("results", {}).get("waste"),
    )


@app.route("/transportation", methods=["GET", "POST"])
def transportation():
    model_ready = transportation_model_ready()
    model_error = MODEL_ERRORS.get("transportation")
    result = None

    if request.method == "POST":
        if not model_ready:
            flash(
                model_error or "Transportation model is not connected.",
                "error",
            )
        else:
            try:
                input_frame = prepare_transportation_input()
                result = run_transportation_prediction(input_frame)

                results = session.get("results", {})
                results["transportation"] = result
                session["results"] = results
                session.modified = True
            except Exception as error:
                model_error = f"{type(error).__name__}: {error}"
                flash(f"Prediction failed: {error}", "error")

    return render_template(
        "transportation.html",
        model_ready=model_ready,
        model_error=model_error,
        result=result,
    )


@app.route("/energy")
def energy():
    return render_template(
        "module_placeholder.html",
        module="Energy",
        module_key="energy",
        icon="⚡",
        model_status=get_model_status(),
        model_error=MODEL_ERRORS.get("energy"),
    )


@app.route("/public-services")
def public_services():
    return render_template(
        "module_placeholder.html",
        module="Public Services",
        module_key="governance",
        icon="🏛️",
        model_status=get_model_status(),
        model_error=MODEL_ERRORS.get("governance"),
    )


@app.route("/waste")
def waste():
    return render_template(
        "waste.html",
        model_ready=model_found("waste"),
        model_error=MODEL_ERRORS.get("waste"),
        result=session.get("results", {}).get("waste"),
        months=list(enumerate(
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December",
            ],
            start=1,
        )),
    )


@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    return render_template("advisor.html", messages=[])


@app.route("/reports")
def reports():
    results = session.get("results", {})
    return render_template(
        "reports.html",
        profile=session.get("profile", default_profile()),
        results=results,
        waste=results.get("waste"),
    )


@app.route("/profile")
def profile():
    profile_data = session.get("profile", default_profile())
    return render_template(
        "profile.html",
        profile=profile_data,
        user_name=session.get("user_name", "City Manager"),
        user_email=session.get("user_email", "demo@city.gov"),
    )


@app.route("/city-profile", methods=["GET", "POST"])
def city_profile():
    profile_data = session.get("profile", default_profile())

    if request.method == "POST":
        profile_data = {
            "city_name": request.form.get("city_name", "London"),
            "country": request.form.get(
                "country",
                "United Kingdom",
            ),
            "role": request.form.get("role", "City Manager"),
            "population": int(
                request.form.get("population", 0) or 0
            ),
            "districts": int(
                request.form.get("districts", 1) or 1
            ),
        }
        session["profile"] = profile_data
        flash("City profile updated.", "success")
        return redirect(url_for("dashboard"))

    return render_template(
        "city_profile.html",
        profile=profile_data,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["logged_in"] = True
        session["user_name"] = request.form.get(
            "name",
            "City Manager",
        )
        session["user_email"] = request.form.get(
            "email",
            "demo@city.gov",
        )
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard"))


@app.route("/model-health")
def model_health():
    health: dict[str, Any] = {}

    for module_name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name
        health[module_name] = {
            "file": f"models/{file_name}",
            "file_exists": model_path.exists(),
            "joblib_loaded": MODEL_BUNDLES.get(module_name) is not None,
            "model_connected": model_found(module_name),
            "loaded_type": (
                type(MODEL_BUNDLES[module_name]).__name__
                if MODEL_BUNDLES.get(module_name) is not None
                else None
            ),
            "error": MODEL_ERRORS.get(module_name),
        }

    return health


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
