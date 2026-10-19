# RNAFunGOA

Gene Ontology (GO) function prediction for long non-coding RNAs (lncRNAs).

## Files

- `main.py`: Model training and evaluation.
- `model_dag.py`: GO function prediction model.
- `utils.py`: Data loading, loss computation, and output utilities.
- `evaluation.py`: Evaluation metrics.
- `lnc_main.py`: Optional lncRNA feature training and export.
- `lnc_model.py`: lncRNA feature extraction model.
- `requirements.txt`: Python dependencies.
- `data/`: GO ontology, graph features, training data, and test data.

## Training

The default training command uses the fixed random seed and a random 15% validation split.

Run from this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```
