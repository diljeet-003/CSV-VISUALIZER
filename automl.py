from pycaret.regression import (
    setup,
    compare_models,
    pull,
    predict_model
)

from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare_models,
    pull as clf_pull,
    predict_model as clf_predict_model
)

import pandas as pd

# ----------------------------------------
# REGRESSION AUTOML
# ----------------------------------------

def run_regression(df, target_column):

    reg_setup = setup(
        data=df,
        target=target_column,
        session_id=123,
        verbose=False
    )

    best_model = compare_models()

    results = pull()

    predictions = predict_model(best_model)

    return best_model, results, predictions

# ----------------------------------------
# CLASSIFICATION AUTOML
# ----------------------------------------

def run_classification(df, target_column):

    clf_setup(
        data=df,
        target=target_column,
        session_id=123,
        verbose=False
    )

    best_model = clf_compare_models()

    results = clf_pull()

    predictions = clf_predict_model(best_model)

    return best_model, results, predictions