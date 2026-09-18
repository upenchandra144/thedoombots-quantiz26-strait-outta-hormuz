# HORMUZ WAR ROOM — Quantiz'26 Round 2

Interactive Streamlit decision-support system for the fictional Hormuz disruption case.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- `app.py` — branded landing / system entry point
- `pages/` — four War Room views
- `src/data_loader.py` — centralized read-only Excel loader
- `src/metrics.py` — reusable analytical definitions
- `src/decision_rules.py` — explainable deterministic shipment decision engine
- `src/scenario_engine.py` — explicit what-if scenario calculations
- `src/components.py` — reusable War Room UI components
- `data/raw/` — source workbook

## Methodology guardrails

- Uses the actual 243 shipment records in the workbook.
- Uses 80% DIFOT as the service-compliance benchmark.
- Treats Direct as a pre-blockade benchmark, not guaranteed future capacity.
- Uses observed / modeled / benchmark terminology rather than causal claims.
- Does not use `Customer_Since` as a decision variable.
- Does not require ML for the core recommendation logic.
- The source workbook is never written back by the application.

## Streamlit Community Cloud

Push the repository to GitHub with `requirements.txt`, select `app.py` as the main file, and deploy.
