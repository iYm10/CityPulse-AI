import calendar
import html
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-on-render")

MODEL_FILES = {
    "transportation": "transportation_bundle.joblib",
    "energy": "energy_bundle.joblib",
    "governance": "governance_bundle.joblib",
    "waste": "waste_bundle.joblib",
}

MODEL_BUNDLES = {}
MODEL_ERRORS = {}


def load_models():
    for module_name, file_name in MODEL_FILES.items():
        path = MODELS_DIR / file_name
        if not path.exists():
            MODEL_BUNDLES[module_name] = None
            MODEL_ERRORS[module_name] = (
                f"File not found: models/{file_name}"
            )
            print(
                f"[MODEL] {module_name}: file not found -> {path}",
                flush=True,
            )
            continue

        try:
            MODEL_BUNDLES[module_name] = joblib.load(path)
            MODEL_ERRORS[module_name] = None
            print(
                f"[MODEL] {module_name}: loaded successfully",
                flush=True,
            )
        except Exception as error:
            MODEL_BUNDLES[module_name] = None
            MODEL_ERRORS[module_name] = (
                f"{type(error).__name__}: {error}"
            )
            print(
                f"[MODEL] {module_name}: load failed -> "
                f"{type(error).__name__}: {error}",
                flush=True,
            )


load_models()


def logged_in():
    return bool(session.get("logged_in"))


def profile_ready():
    return bool(session.get("city_profile"))


def model_found(module_name):
    return MODEL_BUNDLES.get(module_name) is not None


def active_results():
    result_keys = [
        "transportation_result",
        "energy_result",
        "governance_result",
        "waste_result",
    ]
    return [session[key] for key in result_keys if session.get(key)]



def get_model_bundle(module_name):
    raw = MODEL_BUNDLES.get(module_name)

    if raw is None:
        return None

    if isinstance(raw, dict):
        model = None

        for key in (
            "model",
            "pipeline",
            "estimator",
            "best_model",
            "best_estimator",
            "best_estimator_",
            "regressor",
            "classifier",
            "predictor",
            "final_model",
            "trained_model",
        ):
            candidate = raw.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                model = candidate
                break

        if model is None:
            return None

        bundle = dict(raw)
        bundle["model"] = model
        return bundle

    if hasattr(raw, "predict"):
        return {
            "model": raw,
            "feature_columns": list(
                getattr(raw, "feature_names_in_", [])
            ),
            "model_name": raw.__class__.__name__,
            "task_type": "auto",
        }

    return None


def get_bundle_features(bundle):
    features = bundle.get("feature_columns") or bundle.get("features") or []
    if not features:
        features = list(getattr(bundle["model"], "feature_names_in_", []))
    return [str(feature) for feature in features]


def infer_field_schema(feature_name, bundle):
    schemas = bundle.get("input_schema", {}) or {}
    categories = bundle.get("categorical_options", {}) or {}
    defaults = bundle.get("defaults", {}) or {}

    schema = dict(schemas.get(feature_name, {}) or {})
    options = schema.get("options") or categories.get(feature_name) or []
    field_type = "select" if options else schema.get("type", "number")

    return {
        "name": feature_name,
        "label": schema.get("label", feature_name.replace("_", " ").title()),
        "type": field_type,
        "default": schema.get("default", defaults.get(feature_name, "")),
        "options": options,
        "min": schema.get("min"),
        "max": schema.get("max"),
        "step": schema.get("step", 1),
        "help": schema.get("help", ""),
    }


def convert_form_value(raw_value, field):
    if field["type"] == "integer":
        return int(float(raw_value))
    if field["type"] == "number":
        return float(raw_value)
    return raw_value


def prepare_generic_input(module_name, form_data):
    bundle = get_model_bundle(module_name)
    if not bundle:
        raise ValueError(f"{module_name.title()} model bundle is not available.")

    features = get_bundle_features(bundle)

    if features:
        row = {}
        for feature in features:
            field = infer_field_schema(feature, bundle)
            raw_value = form_data.get(feature, "")
            if raw_value == "":
                raw_value = field["default"]
            if raw_value == "":
                raise ValueError(f'Enter a value for "{field["label"]}".')
            row[feature] = convert_form_value(raw_value, field)

        return pd.DataFrame([row], columns=features), row

    raw_json = form_data.get("raw_json", "").strip()
    if not raw_json:
        raise ValueError(
            "This bundle does not expose feature names. "
            "Enter one JSON object containing the exact model inputs."
        )

    row = json.loads(raw_json)
    if not isinstance(row, dict):
        raise ValueError("The JSON input must be one object.")

    return pd.DataFrame([row]), row


def run_generic_prediction(module_name, form_data):
    bundle = get_model_bundle(module_name)
    if not bundle:
        raise ValueError(f"{module_name.title()} model is not available.")

    model = bundle["model"]
    prepared_data, submitted_values = prepare_generic_input(module_name, form_data)
    prediction = model.predict(prepared_data)[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(max(model.predict_proba(prepared_data)[0])) * 100
        except Exception:
            confidence = None

    labels = bundle.get("class_labels") or bundle.get("label_mapping") or {}
    displayed_prediction = labels.get(
        prediction,
        labels.get(str(prediction), prediction),
    )

    return {
        "module": module_name,
        "module_title": {
            "transportation": "Transportation",
            "energy": "Energy",
            "governance": "Public Services",
        }.get(module_name, module_name.title()),
        "prediction": prediction,
        "displayed_prediction": displayed_prediction,
        "confidence": confidence,
        "target_name": bundle.get("target_name", "Prediction"),
        "model_name": bundle.get("model_name", model.__class__.__name__),
        "task_type": bundle.get("task_type", "auto"),
        "inputs": submitted_values,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


GENERIC_MODEL_TEMPLATE = """
{% extends "base.html" %}
{% block title %}{{ module_title }} | CityPulse AI{% endblock %}
{% block content %}
<section class="page-head">
    <div>
        <span class="eyebrow dark">LIVE MODEL</span>
        <h1>{{ icon }} {{ module_title }}</h1>
        <p>{{ subtitle }}</p>
    </div>
    <span class="chip {{ 'live' if model_ready else 'waiting' }}">
        {{ "Model connected" if model_ready else "Model not connected" }}
    </span>
</section>

{% if not model_ready %}
<section class="empty-state">
    <div>📦</div>
    <h3>Add the {{ module_title }} model bundle</h3>
    <p>Confirm that <code>{{ model_file }}</code> exists inside the models folder.</p>
    {% if model_error %}<pre>{{ model_error }}</pre>{% endif %}
</section>
{% else %}
<form class="panel form-grid" method="post">
    {% if fields %}
        {% for field in fields %}
        <label>
            {{ field.label }}
            {% if field.type == "select" %}
                <select name="{{ field.name }}" required>
                    {% for option in field.options %}
                    <option value="{{ option }}"
                        {% if field.default|string == option|string %}selected{% endif %}>
                        {{ option }}
                    </option>
                    {% endfor %}
                </select>
            {% elif field.type == "text" %}
                <input type="text" name="{{ field.name }}" value="{{ field.default }}" required>
            {% else %}
                <input
                    type="number"
                    name="{{ field.name }}"
                    value="{{ field.default }}"
                    step="{{ field.step }}"
                    {% if field.min is not none %}min="{{ field.min }}"{% endif %}
                    {% if field.max is not none %}max="{{ field.max }}"{% endif %}
                    required
                >
            {% endif %}
            {% if field.help %}<small>{{ field.help }}</small>{% endif %}
        </label>
        {% endfor %}
    {% else %}
        <label class="full">
            Model inputs as JSON
            <textarea
                name="raw_json"
                rows="10"
                placeholder='{"feature_1": 10, "feature_2": "value"}'
                required
            ></textarea>
            <small>
                This bundle does not expose feature_columns. Enter one JSON object
                using the exact input names expected by the model.
            </small>
        </label>
    {% endif %}
    <button class="full" type="submit">Generate prediction</button>
</form>
{% endif %}

{% if result %}
<section class="decision-card">
    <span>{{ result.target_name|upper }}</span>
    <h2>{{ result.displayed_prediction }}</h2>
    <p>
        Generated using {{ result.model_name }}
        {% if result.confidence is not none %}
            · Confidence {{ "%.1f"|format(result.confidence) }}%
        {% endif %}
    </p>
</section>

<section class="kpi-grid">
    <article class="kpi">
        <span>Prediction</span>
        <strong>{{ result.displayed_prediction }}</strong>
        <small>{{ result.target_name }}</small>
    </article>
    <article class="kpi">
        <span>Model</span>
        <strong>{{ result.model_name }}</strong>
        <small>{{ result.task_type }}</small>
    </article>
    <article class="kpi">
        <span>Confidence</span>
        <strong>
            {% if result.confidence is not none %}
                {{ "%.1f"|format(result.confidence) }}%
            {% else %}
                N/A
            {% endif %}
        </strong>
        <small>Available for classifiers with predict_proba</small>
    </article>
    <article class="kpi">
        <span>Generated</span>
        <strong>{{ result.created_at[-5:] }}</strong>
        <small>{{ result.created_at[:10] }}</small>
    </article>
</section>

<section class="panel">
    <h3>Submitted inputs</h3>
    <div class="two-grid">
        {% for name, value in result.inputs.items() %}
        <p><strong>{{ name|replace('_', ' ')|title }}:</strong> {{ value }}</p>
        {% endfor %}
    </div>
</section>
{% endif %}
{% endblock %}
"""


def generic_model_page(module_name, module_title, icon, subtitle):
    if not logged_in():
        return redirect(url_for("login"))

    if not profile_ready():
        return redirect(url_for("city_profile"))

    bundle = get_model_bundle(module_name)
    fields = []

    if bundle:
        fields = [
            infer_field_schema(feature, bundle)
            for feature in get_bundle_features(bundle)
        ]

    result_key = f"{module_name}_result"
    result = session.get(result_key)

    if request.method == "POST":
        try:
            result = run_generic_prediction(module_name, request.form)
            session[result_key] = result

            history = session.get("prediction_history", [])
            history.append(result)
            session["prediction_history"] = history[-20:]

            flash(f"{module_title} prediction generated successfully.", "success")
        except Exception as error:
            flash(str(error), "error")

    return render_template_string(
        GENERIC_MODEL_TEMPLATE,
        module_name=module_name,
        module_title=module_title,
        icon=icon,
        subtitle=subtitle,
        model_ready=bundle is not None,
        model_file=MODEL_FILES[module_name],
        model_error=MODEL_ERRORS.get(module_name),
        fields=fields,
        result=result,
    )



def get_transportation_bundle():
    bundle = MODEL_BUNDLES.get("transportation")

    if not isinstance(bundle, dict):
        return None

    required_keys = {
        "stage1_model",
        "stage2_model",
        "feature_columns",
    }

    if not required_keys.issubset(bundle):
        return None

    if not hasattr(bundle["stage1_model"], "predict"):
        return None

    if not hasattr(bundle["stage2_model"], "predict"):
        return None

    return bundle


def transportation_model_ready():
    bundle = get_transportation_bundle()
    return bundle is not None and bool(bundle.get("feature_columns"))


def prepare_transportation_input(form_data):
    bundle = get_transportation_bundle()

    if not transportation_model_ready():
        raise ValueError(
            "Transportation bundle must contain stage1_model, "
            "stage2_model, and feature_columns."
        )

    feature_columns = [str(column) for column in bundle["feature_columns"]]
    row = {feature: 0 for feature in feature_columns}

    numeric_defaults = {
        "air_co": 0,
        "air_nox": 40,
        "air_no2": 30,
        "air_no": 10,
        "air_o3": 40,
        "air_air_temp": 15,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "weather_tavg": 15,
        "weather_tmin": 10,
        "weather_tmax": 20,
        "weather_prcp": 0,
        "weather_wdir": 180,
        "weather_wspd": 10,
        "weather_pres": 1013,
        "bike_cnt": 1000,
        "bike_t1": 15,
        "bike_t2": 14,
        "bike_hum": 70,
        "bike_wind_speed": 10,
        "bike_weather_code": 1,
        "bike_is_holiday": 0,
        "year": 2026,
        "month": 1,
        "hour": 12,
        "day_of_week": 0,
        "is_weekend": 0,
    }

    integer_features = {
        "bike_is_holiday",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
    }

    submitted_values = {}

    for feature_name, default_value in numeric_defaults.items():
        raw_value = form_data.get(feature_name, "")
        if raw_value is None or str(raw_value).strip() == "":
            raw_value = default_value

        value = (
            int(float(raw_value))
            if feature_name in integer_features
            else float(raw_value)
        )

        submitted_values[feature_name] = value

        if feature_name in row:
            row[feature_name] = value

    categorical_values = {
        "site": form_data.get("site", "London Bloomsbury"),
        "code": form_data.get("code", "CLL2"),
        "site_type": form_data.get("site_type", "Urban Background"),
        "season": form_data.get("season", "Winter"),
    }

    for prefix, selected_value in categorical_values.items():
        selected_column = f"{prefix}_{selected_value}"

        if selected_column not in row:
            available_values = [
                column[len(prefix) + 1:]
                for column in feature_columns
                if column.startswith(f"{prefix}_")
            ]
            raise ValueError(
                f"Unsupported {prefix} value: {selected_value}. "
                f"Available values: {', '.join(available_values)}"
            )

        row[selected_column] = 1
        submitted_values[prefix] = selected_value

    prepared_data = pd.DataFrame(
        [row],
        columns=feature_columns,
    ).apply(pd.to_numeric, errors="raise")

    return prepared_data, submitted_values


def run_transportation_prediction(form_data):
    bundle = get_transportation_bundle()

    if not transportation_model_ready():
        raise ValueError("Transportation model is not connected.")

    prepared_data, submitted_values = prepare_transportation_input(form_data)

    stage1_model = bundle["stage1_model"]
    stage2_model = bundle["stage2_model"]
    threshold = float(bundle.get("classification_threshold", 0.5))

    if hasattr(stage1_model, "predict_proba"):
        probability = float(
            stage1_model.predict_proba(prepared_data)[0][-1]
        )
    else:
        probability = float(stage1_model.predict(prepared_data)[0])
        probability = max(0.0, min(1.0, probability))

    collision_expected = probability >= threshold

    if collision_expected:
        predicted_count = float(stage2_model.predict(prepared_data)[0])
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

    return {
        "module": "Transportation",
        "prediction": predicted_count,
        "probability": probability,
        "probability_percent": probability_percent,
        "collision_expected": collision_expected,
        "predicted_collision_count": predicted_count,
        "threshold": threshold,
        "risk_level": risk_level,
        "status": status,
        "headline": (
            f"Approximately {predicted_count:.1f} collisions expected"
            if collision_expected
            else "Low probability of collision activity"
        ),
        "target_name": bundle.get("target", "collision_count"),
        "model_name": bundle.get("model_name", "Two-stage LightGBM"),
        "task_type": bundle.get("task", "hurdle_regression"),
        "unit": bundle.get("unit", "collisions"),
        "inputs": submitted_values,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }



def get_energy_bundle():
    bundle = MODEL_BUNDLES.get("energy")

    if not isinstance(bundle, dict):
        return None

    required_keys = {
        "model",
        "feature_columns",
        "category_maps",
    }

    if not required_keys.issubset(bundle):
        return None

    if not hasattr(bundle["model"], "predict"):
        return None

    return bundle


def energy_model_ready():
    bundle = get_energy_bundle()

    return (
        bundle is not None
        and list(bundle.get("feature_columns", []))
        == [
            "building_id",
            "site_id",
            "primaryspaceusage",
            "sqm",
            "airTemperature",
            "dewTemperature",
            "seaLvlPressure",
            "windDirection",
            "windSpeed",
            "hour",
            "day_of_week",
            "is_weekend",
            "temperature_squared",
            "month",
        ]
    )


def prepare_energy_input(form_data):
    bundle = get_energy_bundle()

    if not energy_model_ready():
        raise ValueError(
            "Energy bundle is unavailable or does not contain "
            "the exact 14 training features."
        )

    category_maps = bundle["category_maps"]

    building_name = form_data.get("building_id", "").strip()
    site_name = form_data.get("site_id", "").strip()
    usage_name = form_data.get("primaryspaceusage", "").strip()

    for field_name, selected_value in (
        ("building_id", building_name),
        ("site_id", site_name),
        ("primaryspaceusage", usage_name),
    ):
        if selected_value not in category_maps[field_name]:
            raise ValueError(
                f"Unsupported value for {field_name}: {selected_value}"
            )

    air_temperature = float(form_data["airTemperature"])

    row = {
        "building_id": category_maps["building_id"][building_name],
        "site_id": category_maps["site_id"][site_name],
        "primaryspaceusage": category_maps[
            "primaryspaceusage"
        ][usage_name],
        "sqm": float(form_data["sqm"]),
        "airTemperature": air_temperature,
        "dewTemperature": float(form_data["dewTemperature"]),
        "seaLvlPressure": float(form_data["seaLvlPressure"]),
        "windDirection": float(form_data["windDirection"]),
        "windSpeed": float(form_data["windSpeed"]),
        "hour": int(form_data["hour"]),
        "day_of_week": int(form_data["day_of_week"]),
        "is_weekend": int(form_data["is_weekend"]),
        "temperature_squared": air_temperature ** 2,
        "month": int(form_data["month"]),
    }

    features = bundle["feature_columns"]
    prepared_data = pd.DataFrame([row], columns=features)

    submitted_values = {
        "Building": building_name,
        "Site": site_name,
        "Primary space usage": usage_name,
        "Floor area (sqm)": row["sqm"],
        "Air temperature": row["airTemperature"],
        "Dew temperature": row["dewTemperature"],
        "Sea-level pressure": row["seaLvlPressure"],
        "Wind direction": row["windDirection"],
        "Wind speed": row["windSpeed"],
        "Hour": row["hour"],
        "Day of week": row["day_of_week"],
        "Weekend": "Yes" if row["is_weekend"] else "No",
        "Month": row["month"],
    }

    return prepared_data, submitted_values


def run_energy_prediction(form_data):
    bundle = get_energy_bundle()

    if not energy_model_ready():
        raise ValueError("Energy model is not connected.")

    prepared_data, submitted_values = prepare_energy_input(form_data)
    prediction = float(bundle["model"].predict(prepared_data)[0])
    prediction = max(0.0, prediction)

    if prediction >= 1000:
        level = "High"
        status = "Prepare for elevated consumption"
    elif prediction >= 400:
        level = "Moderate"
        status = "Monitor demand and efficiency"
    else:
        level = "Low"
        status = "Normal consumption outlook"

    return {
        "module": "Energy",
        "prediction": prediction,
        "displayed_prediction": f"{prediction:,.2f}",
        "target_name": bundle.get(
            "target",
            "electricity_consumption",
        ),
        "model_name": bundle.get(
            "model_name",
            bundle["model"].__class__.__name__,
        ),
        "task_type": bundle.get("task", "regression"),
        "level": level,
        "status": status,
        "inputs": submitted_values,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }



def get_waste_bundle():
    bundle = MODEL_BUNDLES.get("waste")
    return bundle if isinstance(bundle, dict) else None


def prepare_waste_input(
    borough,
    district,
    year,
    month_number,
    last_month,
    two_months,
):
    bundle = get_waste_bundle()
    if not bundle:
        raise ValueError("Waste model bundle is not available.")

    features = bundle.get("feature_columns", [])
    if not features:
        raise ValueError("The waste bundle does not contain feature_columns.")

    row = {feature: 0 for feature in features}
    numeric_values = {
        "year": year,
        "month_number": month_number,
        "waste_last_month": last_month,
        "waste_2_months_ago": two_months,
    }

    for feature in features:
        name = str(feature)

        if name in numeric_values:
            row[feature] = numeric_values[name]

        if name == f"borough_{borough}":
            row[feature] = 1

        if name in {
            f"communitydistrict_{district}",
            f"communitydistrict_{float(district)}",
            f"communitydistrict_{str(district)}",
        }:
            row[feature] = 1

    return pd.DataFrame([row], columns=features)


def run_waste_prediction(
    borough,
    district,
    year,
    month_number,
    last_month,
    two_months,
):
    bundle = get_waste_bundle()
    if not bundle or bundle.get("model") is None:
        raise ValueError("Waste model is not available.")

    prepared_data = prepare_waste_input(
        borough=borough,
        district=district,
        year=year,
        month_number=month_number,
        last_month=last_month,
        two_months=two_months,
    )

    prediction = float(bundle["model"].predict(prepared_data)[0])
    difference = prediction - last_month
    change_percent = (difference / last_month * 100) if last_month else 0.0

    if change_percent >= 10:
        priority = "High"
        status = "Prepare for higher demand"
        headline = "Collection demand is expected to rise noticeably."
        summary = (
            "The selected district may need earlier operational preparation "
            "before the forecast month begins."
        )
        actions = [
            ["Review collection capacity", "Check whether the current plan can absorb the expected increase."],
            ["Confirm workforce coverage", "Review shifts and availability for the selected district."],
            ["Monitor the first cycle", "Compare actual volume with the forecast and adjust quickly."],
        ]
    elif change_percent >= 5:
        priority = "Medium"
        status = "Needs attention"
        headline = "Waste demand may increase moderately."
        summary = "Review the current plan and prepare a small operational buffer."
        actions = [
            ["Review the monthly schedule", "Check collection timing for the selected district."],
            ["Prepare a small buffer", "Keep additional capacity available if actual volume rises."],
            ["Track the next update", "Run the forecast again when new monthly data becomes available."],
        ]
    elif change_percent <= -10:
        priority = "Low"
        status = "Lower demand expected"
        headline = "Waste demand is expected to decrease."
        summary = "There may be an opportunity to optimize resources while maintaining service quality."
        actions = [
            ["Keep service quality stable", "Do not reduce essential coverage based on one forecast alone."],
            ["Review resource efficiency", "Check whether some capacity can support another nearby area."],
            ["Confirm with actual data", "Compare the forecast with the first collection cycle."],
        ]
    else:
        priority = "Low"
        status = "Stable outlook"
        headline = "Waste demand is expected to remain close to last month."
        summary = "The current collection plan appears suitable, with no urgent expansion indicated."
        actions = [
            ["Maintain the current plan", "Continue the normal collection schedule for this district."],
            ["Watch for local events", "Adjust only if events or service disruptions change demand."],
            ["Update next month", "Use the next actual value to improve the following forecast."],
        ]

    return {
        "module": "Waste Management",
        "borough": borough,
        "district": int(district),
        "year": int(year),
        "month": int(month_number),
        "month_name": calendar.month_name[int(month_number)],
        "prediction": prediction,
        "difference": difference,
        "change_percent": change_percent,
        "last_month": float(last_month),
        "two_months": float(two_months),
        "priority": priority,
        "status": status,
        "headline": headline,
        "summary": summary,
        "actions": actions,
        "model_name": bundle.get("model_name", "Waste Forecast Model"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def rule_based_advisor(question):
    waste = session.get("waste_result")
    city_name = session.get("city_profile", {}).get("city_name", "the city")
    q = question.lower()

    if not active_results():
        return (
            "There are no completed assessments yet. Start with one smart-city area, "
            "then I can explain the result and turn it into an action plan."
        )

    if waste:
        area = f'{waste["borough"]}, District {waste["district"]}'
        period = f'{waste["month_name"]} {waste["year"]}'

        if any(word in q for word in ["action", "plan", "prepare", "next step"]):
            steps = "\n".join(
                f"{i}. {title}: {text}"
                for i, (title, text) in enumerate(waste["actions"], start=1)
            )
            return f"For {area} in {period}, the outlook is {waste['status']}.\n\n{steps}"

        if "why" in q or "reason" in q:
            return (
                f"The forecast for {area} is {waste['prediction']:,.0f} tons, "
                f"which is {waste['change_percent']:+.1f}% compared with last month."
            )

        if any(word in q for word in ["priority", "main problem", "most important"]):
            return (
                f"The latest available priority for {city_name} is the waste outlook in {area}. "
                f"The planning level is {waste['priority']}."
            )

        return (
            f"The latest waste forecast for {area} is {waste['prediction']:,.0f} tons "
            f"for {period}. The outlook is {waste['status']}. {waste['summary']}"
        )

    return "Completed assessments were found, but more detail is required."


def ask_ai(question):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return rule_based_advisor(question), "Smart guidance"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        context = json.dumps(
            {
                "city_profile": session.get("city_profile", {}),
                "latest_results": active_results(),
            },
            ensure_ascii=False,
        )

        response = client.responses.create(
            model=model_name,
            instructions=(
                "You are CityPulse AI, a city decision-support advisor. "
                "Use only the supplied city profile and model results. "
                "Do not invent budgets, staffing, causes, probabilities, or forecasts. "
                "Answer with: What is happening, Why it matters, Recommended next action."
            ),
            input=f"CITY CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
        )
        return response.output_text, f"AI advisor · {model_name}"
    except Exception:
        return rule_based_advisor(question), "Smart guidance fallback"


@app.context_processor
def inject_globals():
    return {
        "model_status": {name: model_found(name) for name in MODEL_FILES},
        "current_year": datetime.now().year,
    }


@app.route("/")
def index():
    if not logged_in():
        return redirect(url_for("login"))
    if not profile_ready():
        return redirect(url_for("city_profile"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        mode = request.form.get("mode", "signin")
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if mode == "create":
            name = request.form.get("name", "").strip()
            if not name or not email or not password:
                flash("Complete all account fields.", "error")
                return redirect(url_for("login"))
            session["user_name"] = name
        else:
            if not email or not password:
                flash("Enter your email and password.", "error")
                return redirect(url_for("login"))
            session["user_name"] = email.split("@")[0].replace(".", " ").title()

        session["logged_in"] = True
        session["user_email"] = email
        return redirect(url_for("city_profile") if not profile_ready() else url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/city-profile", methods=["GET", "POST"])
def city_profile():
    if not logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":
        city_name = request.form.get("city_name", "").strip()
        country = request.form.get("country", "").strip()

        if not city_name or not country:
            flash("Enter the city name and country.", "error")
            return redirect(url_for("city_profile"))

        session["city_profile"] = {
            "city_name": city_name,
            "country": country,
            "role": request.form.get("role", "City Manager"),
            "population": int(request.form.get("population", 100000)),
            "districts": int(request.form.get("districts", 10)),
            "language": request.form.get("language", "English"),
            "goals": request.form.getlist("goals"),
            "selected_modules": request.form.getlist("selected_modules"),
        }
        return redirect(url_for("dashboard"))

    return render_template("city_profile.html", profile=session.get("city_profile", {}))


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    if not profile_ready():
        return redirect(url_for("city_profile"))

    transportation_result = session.get("transportation_result")
    energy_result = session.get("energy_result")
    governance_result = session.get("governance_result")
    waste_result = session.get("waste_result")

    latest_result = None
    available_results = [
        result
        for result in (
            transportation_result,
            energy_result,
            governance_result,
            waste_result,
        )
        if result
    ]

    if available_results:
        latest_result = max(
            available_results,
            key=lambda result: result.get("created_at", ""),
        )

    return render_template(
        "dashboard.html",
        profile=session["city_profile"],
        user_name=session.get("user_name", "User"),
        results=available_results,
        latest_result=latest_result,
        transportation=transportation_result,
        energy=energy_result,
        governance=governance_result,
        waste=waste_result,
    )


@app.route("/transportation", methods=["GET", "POST"])
def transportation():
    if not logged_in():
        return redirect(url_for("login"))

    if not profile_ready():
        return redirect(url_for("city_profile"))

    result = session.get("transportation_result")
    bundle = get_transportation_bundle()

    feature_columns = (
        [str(column) for column in bundle.get("feature_columns", [])]
        if bundle
        else []
    )

    def category_options(prefix):
        return [
            column[len(prefix) + 1:]
            for column in feature_columns
            if column.startswith(f"{prefix}_")
        ]

    if request.method == "POST":
        try:
            result = run_transportation_prediction(request.form)
            session["transportation_result"] = result

            history = session.get("prediction_history", [])
            history.append(result)
            session["prediction_history"] = history[-20:]

            flash(
                "Transportation assessment generated successfully.",
                "success",
            )
        except Exception as error:
            flash(str(error), "error")

    return render_template(
        "transportation.html",
        result=result,
        model_ready=transportation_model_ready(),
        model_error=MODEL_ERRORS.get("transportation"),
        site_options=category_options("site"),
        code_options=category_options("code"),
        site_type_options=category_options("site_type"),
        season_options=category_options("season"),
    )


@app.route("/energy", methods=["GET", "POST"])
def energy():
    if not logged_in():
        return redirect(url_for("login"))

    if not profile_ready():
        return redirect(url_for("city_profile"))

    bundle = get_energy_bundle()
    result = session.get("energy_result")

    building_options = []
    site_options = []
    usage_options = []

    if bundle:
        category_maps = bundle.get("category_maps", {})
        building_options = list(
            category_maps.get("building_id", {}).keys()
        )
        site_options = list(
            category_maps.get("site_id", {}).keys()
        )
        usage_options = list(
            category_maps.get(
                "primaryspaceusage",
                {},
            ).keys()
        )

    if request.method == "POST":
        try:
            result = run_energy_prediction(request.form)
            session["energy_result"] = result

            history = session.get("prediction_history", [])
            history.append(result)
            session["prediction_history"] = history[-20:]

            flash(
                "Energy forecast generated successfully.",
                "success",
            )
        except Exception as error:
            flash(str(error), "error")

    return render_template(
        "energy.html",
        result=result,
        model_ready=energy_model_ready(),
        model_error=MODEL_ERRORS.get("energy"),
        building_options=building_options,
        site_options=site_options,
        usage_options=usage_options,
    )


@app.route("/public-services", methods=["GET", "POST"])
def public_services():
    return generic_model_page(
        module_name="governance",
        module_title="Public Services",
        icon="🏛️",
        subtitle="Use the connected model to prioritize or assess public-service requests.",
    )


@app.route("/waste", methods=["GET", "POST"])
def waste():
    if not logged_in():
        return redirect(url_for("login"))
    if not profile_ready():
        return redirect(url_for("city_profile"))

    result = session.get("waste_result")

    if request.method == "POST":
        try:
            result = run_waste_prediction(
                borough=request.form["borough"],
                district=int(request.form["district"]),
                year=int(request.form["year"]),
                month_number=int(request.form["month"]),
                last_month=float(request.form["last_month"]),
                two_months=float(request.form["two_months"]),
            )
            session["waste_result"] = result
            history = session.get("prediction_history", [])
            history.append(result)
            session["prediction_history"] = history[-20:]
            flash("Waste plan generated successfully.", "success")
        except Exception as error:
            flash(str(error), "error")

    return render_template(
        "waste.html",
        result=result,
        model_ready=model_found("waste"),
        model_error=MODEL_ERRORS.get("waste"),
        months=list(enumerate(calendar.month_name))[1:],
    )


@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    if not logged_in():
        return redirect(url_for("login"))

    messages = session.get("advisor_messages", [])

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer, source = ask_ai(question)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer, "source": source})
            session["advisor_messages"] = messages[-20:]

    return render_template("advisor.html", messages=messages)


@app.route("/reports")
def reports():
    if not logged_in():
        return redirect(url_for("login"))

    transportation_result = session.get("transportation_result")
    energy_result = session.get("energy_result")
    governance_result = session.get("governance_result")
    waste_result = session.get("waste_result")

    results = [
        result
        for result in (
            transportation_result,
            energy_result,
            governance_result,
            waste_result,
        )
        if result
    ]

    results.sort(
        key=lambda result: result.get("created_at", ""),
        reverse=True,
    )

    return render_template(
        "reports.html",
        profile=session.get("city_profile", {}),
        results=results,
        transportation=transportation_result,
        energy=energy_result,
        governance=governance_result,
        waste=waste_result,
    )


@app.route("/download-report")
def download_report():
    if not logged_in():
        return redirect(url_for("login"))

    profile = session.get("city_profile", {})
    transportation = session.get("transportation_result")
    energy = session.get("energy_result")
    governance = session.get("governance_result")
    waste = session.get("waste_result")

    sections = []

    if transportation:
        sections.append(
            f"""
            <section class="report-section">
                <h2>Transportation Outlook</h2>
                <p><strong>Risk level:</strong>
                    {html.escape(str(transportation.get('risk_level', 'N/A')))}
                </p>
                <p><strong>Collision probability:</strong>
                    {float(transportation.get('probability_percent', 0)):.1f}%
                </p>
                <p><strong>Expected collisions:</strong>
                    {float(transportation.get('predicted_collision_count', 0)):.1f}
                    {html.escape(str(transportation.get('unit', 'collisions')))}
                </p>
                <p><strong>Status:</strong>
                    {html.escape(str(transportation.get('status', '')))}
                </p>
                <p><strong>Model:</strong>
                    {html.escape(str(transportation.get('model_name', '')))}
                </p>
            </section>
            """
        )

    if energy:
        sections.append(
            f"""
            <section class="report-section">
                <h2>Energy Outlook</h2>
                <p><strong>Predicted consumption:</strong>
                    {html.escape(str(energy.get('displayed_prediction', energy.get('prediction', 'N/A'))))}
                </p>
                <p><strong>Demand level:</strong>
                    {html.escape(str(energy.get('level', 'N/A')))}
                </p>
                <p><strong>Status:</strong>
                    {html.escape(str(energy.get('status', '')))}
                </p>
                <p><strong>Target:</strong>
                    {html.escape(str(energy.get('target_name', 'Prediction')))}
                </p>
                <p><strong>Model:</strong>
                    {html.escape(str(energy.get('model_name', '')))}
                </p>
            </section>
            """
        )

    if governance:
        governance_prediction = governance.get(
            "displayed_prediction",
            governance.get("prediction", "N/A"),
        )
        confidence = governance.get("confidence")
        confidence_line = (
            f"<p><strong>Confidence:</strong> {float(confidence):.1f}%</p>"
            if confidence is not None
            else ""
        )

        sections.append(
            f"""
            <section class="report-section">
                <h2>Public Services Outlook</h2>
                <p><strong>Prediction:</strong>
                    {html.escape(str(governance_prediction))}
                </p>
                <p><strong>Target:</strong>
                    {html.escape(str(governance.get('target_name', 'Prediction')))}
                </p>
                {confidence_line}
                <p><strong>Model:</strong>
                    {html.escape(str(governance.get('model_name', '')))}
                </p>
            </section>
            """
        )

    if waste:
        actions = "".join(
            f"<li><strong>{html.escape(title)}</strong>: "
            f"{html.escape(text)}</li>"
            for title, text in waste.get("actions", [])
        )

        sections.append(
            f"""
            <section class="report-section">
                <h2>Waste Management Outlook</h2>
                <p><strong>Area:</strong>
                    {html.escape(str(waste.get('borough', '')))},
                    District {waste.get('district', '')}
                </p>
                <p><strong>Forecast period:</strong>
                    {html.escape(str(waste.get('month_name', '')))}
                    {waste.get('year', '')}
                </p>
                <p><strong>Expected waste:</strong>
                    {float(waste.get('prediction', 0)):,.0f} tons
                </p>
                <p><strong>Change from last month:</strong>
                    {float(waste.get('change_percent', 0)):+.1f}%
                </p>
                <p><strong>Planning status:</strong>
                    {html.escape(str(waste.get('status', '')))}
                </p>
                <ol>{actions}</ol>
            </section>
            """
        )

    if not sections:
        sections.append(
            """
            <section class="report-section">
                <h2>No completed assessments</h2>
                <p>Complete a smart-city assessment before downloading the report.</p>
            </section>
            """
        )

    report_sections = "\n".join(sections)

    report = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>CityPulse AI Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                color: #142033;
                line-height: 1.6;
            }}
            .head {{
                background: #0B2745;
                color: white;
                padding: 28px;
                border-radius: 18px;
            }}
            .meta {{
                background: #F3F6FA;
                padding: 18px;
                border-radius: 14px;
                margin: 18px 0;
            }}
            .report-section {{
                border: 1px solid #DCE4ED;
                border-radius: 16px;
                padding: 22px;
                margin: 18px 0;
            }}
            .report-section h2 {{
                margin-top: 0;
                color: #0B2745;
            }}
        </style>
    </head>
    <body>
        <div class="head">
            <h1>CityPulse AI Executive Report</h1>
            <p>
                {html.escape(profile.get('city_name', 'City'))},
                {html.escape(profile.get('country', ''))}
            </p>
        </div>

        <div class="meta">
            <strong>Prepared:</strong>
            {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
            <strong>City manager:</strong>
            {html.escape(session.get('user_name', ''))}<br>
            <strong>Completed assessments:</strong>
            {len(active_results())}
        </div>

        {report_sections}

        <hr>
        <small>
            Decision support only. Final operational decisions require
            human review.
        </small>
    </body>
    </html>
    """

    filename = (
        f"CityPulse_"
        f"{profile.get('city_name', 'City').replace(' ', '_')}"
        f"_Report.html"
    )

    return send_file(
        BytesIO(report.encode("utf-8")),
        mimetype="text/html",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/profile")
def profile():
    if not logged_in():
        return redirect(url_for("login"))

    return render_template(
        "profile.html",
        profile=session.get("city_profile", {}),
        user_name=session.get("user_name", ""),
        user_email=session.get("user_email", ""),
    )



@app.route("/model-health")
def model_health():
    health = {}

    for module_name, file_name in MODEL_FILES.items():
        path = MODELS_DIR / file_name
        raw_bundle = MODEL_BUNDLES.get(module_name)

        health[module_name] = {
            "file": f"models/{file_name}",
            "file_exists": path.exists(),
            "bundle_loaded": raw_bundle is not None,
            "model_connected": model_found(module_name),
            "loaded_type": (
                type(raw_bundle).__name__
                if raw_bundle is not None
                else None
            ),
            "bundle_keys": (
                list(raw_bundle.keys())
                if isinstance(raw_bundle, dict)
                else []
            ),
            "error": MODEL_ERRORS.get(module_name),
        }

    return health


@app.errorhandler(404)
def page_not_found(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=True,
    )
