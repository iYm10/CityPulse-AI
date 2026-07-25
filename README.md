# CityPulse AI — Flask Edition

A premium Flask-based smart city decision-support platform rebuilt from the Streamlit prototype.

## Included

- Prototype login and account creation
- City workspace profile
- Executive command-center dashboard
- Transportation, energy, public services, and waste workspaces
- Waste model integration using `models/waste_bundle.joblib`
- Rule-based advisor with optional OpenAI integration
- Executive HTML report download
- Responsive premium interface
- Render deployment configuration

## Deploy on Render

Choose **Web Service**.

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
gunicorn app:app
```

## Model files

Place trusted model bundles inside `models/`:

- `waste_bundle.joblib`
- `transportation_bundle.joblib`
- `energy_bundle.joblib`
- `governance_bundle.joblib`

The waste bundle is expected to contain:

- `model`
- `feature_columns`
- optionally `model_name`

## Environment variables

- `SECRET_KEY`: recommended for production
- `OPENAI_API_KEY`: optional
- `OPENAI_MODEL`: optional

## Important

The included authentication is a prototype only. Before real production use, add:

- a database
- password hashing
- CSRF protection
- server-side sessions
- user authorization
