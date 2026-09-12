# pharmaguide

**The invisible med list: a safer medication conversation for African clinics.**

PharmaGuide is an offline-first prototype that combines a patient medication list
with over-the-counter products and locally used herbs. It flags potentially risky
medication interactions and produces a plain-language prompt for a clinician. It
is a decision-support prototype, not medical advice.

## How it works

`medication names -> local SMILES/cache -> fingerprints -> interaction model -> flags`

The interaction model is trained from the curated examples in `train.py`. The
training path is deterministic and does not require downloading DDInter data.
`data/chem.py` uses built-in SMILES and a deterministic fingerprint fallback when
RDKit or a cached PubChem result is unavailable.

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python train.py --quick
uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \\
  -H 'content-type: application/json' \\
  -d '{"medications":["aspirin","warfarin"]}'
curl http://127.0.0.1:8000/herbs
```

`POST /detect` accepts at least two medication names. `POST /detect-herb` also
accepts a manual `name`, optional comma-separated `medications`, and an optional
image upload. Image identification is best-effort: the API uses the local herb
model when its artifacts are present, then an optional vision service, and finally
returns a manual-identification response.

Run the offline test suite with:

```bash
pytest -q
```

## Generated model artifacts

`python train.py --quick` writes these local artifacts:

- `models/interaction_model.joblib`
- `models/severity_labels.json`

They are intentionally ignored by Git. The GitHub Actions workflow runs tests,
trains offline on pushes to `main` or manual dispatch, verifies both artifacts,
and uploads them as a short-lived workflow artifact. `models/.gitkeep` keeps the
directory in the repository.

## Limitations and safety

This is a prototype, not a substitute for a clinician or pharmacist. It does not
prove that a product is safe, may miss brand names, herbal ingredients, OCR
errors, regional products, or interactions absent from its training examples.
Never stop or change a medicine based only on this output; validate every flag
with a qualified professional.

DDInter and PubChem have their own terms and provenance. Review upstream sources
before redistribution or clinical use.
