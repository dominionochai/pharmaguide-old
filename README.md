# pharmaguide

PharmaGuide is an offline-first medicine conversation service for African clinics. It combines a patient medication list with locally used herbs and a curated interaction catalog, then returns plain-language prompts for a clinician. It is decision support, not medical advice.

## How it works

`medicine names -> case-insensitive catalog lookup -> matched medicine and herb items -> database interaction flags -> optional trained severity model`

The API uses the local catalog first. A trained interaction model is used when `models/interaction_model.joblib` is present; otherwise `/detect` continues in database-only mode.

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --quick
uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \
  -H 'content-type: application/json' \
  -d '{"medications":["amlodipine","simvastatin"]}'
```

`POST /detect` accepts at least two case-insensitive medicine or herb names and returns matched items, interaction flags, their severity and confidence, the active detection mode, and a tell-your-doctor sheet. `POST /detect-herb` accepts a manual herb name, an optional comma-separated medication list, and an optional image upload.

## Generated artifacts

`python train.py --quick` writes `models/interaction_model.joblib` and `models/severity_labels.json`. The repository workflow can rebuild these artifacts when needed; the API remains usable without them.

## Safety

The catalog cannot establish that a product is safe and may not cover brand names, formulation differences, or interactions absent from its sources. Do not stop or change a medicine based only on this output. Have every flag reviewed by a qualified clinician or pharmacist.
