"""Central filesystem layout for module_4, anchored to the folder via __file__ so
every script resolves the same paths regardless of the current working directory.

    module_4/
      src/         <- this file
      data/        holdout inputs
      artifacts/   trained models + cached predictions
      notebooks/
      plots/
      predictions/ named deliverables (+ exploration/ scratch CSVs)
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # module_4/

DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(ROOT, "plots")
PREDICTIONS = os.path.join(ROOT, "predictions")
EXPLORATION = os.path.join(PREDICTIONS, "exploration")
NOTEBOOKS = os.path.join(ROOT, "notebooks")

# Data inputs
MINI_HOLDOUT = os.path.join(DATA, "biking_holdout_test_mini.csv")
BIG_HOLDOUT = os.path.join(DATA, "bikes_december.csv")

# Trained-model directories / files
MODELS_V1 = os.path.join(ARTIFACTS, "models")
MODELS_V2 = os.path.join(ARTIFACTS, "models_v2")
MODELS_V3 = os.path.join(ARTIFACTS, "models_v3")
BEST_MODEL = os.path.join(ARTIFACTS, "best_model")
BEST_MODEL_V3 = os.path.join(ARTIFACTS, "best_model_v3")
BASE_PREDS = os.path.join(ARTIFACTS, "base_preds.npz")

TOML = os.path.join(ROOT, "module4.toml")
ANSWERS_URL = (
    "https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/"
    "biking_holdout_test_mini_answers.csv"
)


def expl(name: str) -> str:
    """Path to a scratch <name>-predictions.csv under predictions/exploration/."""
    return os.path.join(EXPLORATION, name)
