# pharmaguide

**the invisible med list African doctors never get to see**

A Python prototype: a patient snaps photos of packs, the app builds the full med list (western + OTC + herbal), flags dangerous interactions with confidence, and prints a plain-language tell-your-doctor sheet.

## How it works

`photo -> OCR glue -> med list -> fingerprint interaction model -> flags`

This repository supplies the interaction-model and API pieces. OCR glue can pass recognized names to a `POST /detect`; a production photo pipeline would add image capture and OCR.

## The African medicine-bag story

A medicine bag in an African clinic may contain a prescription, an OTC painkiller, a supplement, and a locally used herbal preparation. A clinician often sees only the medicine the patient remembers to mention. pharmaguide is designed around the real bag: collect every pack, preserve uncertainty, and make the complete list easy to discuss with a doctor.

## ML artifact

There is **ONE trained ML model**: an interaction severity classifier trained on DDInter. `train.py` maps names to PubChem canonical SMILES, computes RDKit Morgan fingerprints (radius 2, 1024 bits), XORS the two fingerprints, and trains a CPU-only class-balanced multi-class RandomForest. The artifact is `models/interaction_model.joblib`, with labels in `models/severity_labels.json`.

## Quickstart

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python data/fetch_ddinter.py
python train.py
uvicorn api.main:app --reload
curl -X POST http://127.0.0.1:8000/detect -H 'content-type: application/json' -d '{"medications":["aspirin","warfarin"]}'
pytest -q
```

`GET /health` works before training and reports whether the model is loaded. `api/demo_sheet.py` renders a short English + simple pidgin tell-your-doctor line for a flag.

## Limitations

This is a prototype, not medical advice. It is not a substitute for a clinician or pharmacist, does not prove that a product is safe, and may miss brand names, herbal ingredients, OCR errors, regional products, and interactions absent from DDInter. Never stop or change a medicine based on this output. Validate every flag with a qualified professional.

## License and data

DDInter and PubChem have their own terms and provenance. Check upstream sources before redistribution or clinical use.

## Train in the cloud

Train in the cloud: push to GitHub → Actions → Train PharmaGuide models → download pharmaguide-models artifact
