# Model card

## Intended use

Compare shot quality within exploratory opposition or team analysis. The probability is a descriptive aid, not a player-evaluation score and not a live match recommendation.

## Model

Regularised logistic regression using location-derived distance and angle plus pressure, first-time, body-part, technique, and open-play indicators. Preprocessing and classification are one fitted pipeline.

## Validation

All shots from the same match stay in the same fold. This prevents near-duplicate match context leaking across training and validation. The CLI reports Brier score, log loss, ROC AUC, and five-bin expected calibration error.

## Known limitations

- The checked-in demo uses a deliberately small open-data sample, so uncertainty is material.
- Event data does not encode every defender's position for every shot.
- Rare goals make calibration estimates unstable on a small sample; the model therefore avoids class weighting and reports calibration error explicitly.
- A production model should use more seasons, hierarchical effects, calibration on held-out seasons, and subgroup checks.
