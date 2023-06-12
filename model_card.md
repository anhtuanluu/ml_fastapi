# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier from the sklearn library.

## Intended Use

Predict the income that exceeds $50K/yr.

## Training Data

dataset: [dataset](https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data

Break dataset into a train and evaluation set with 80:20 ratio

## Metrics

The model is evaluated on the metrics Precision, Recall and Fbeta

## Ethical Considerations

The data contains attributes about sex, race. Should consider how the model performs across groups.

## Caveats and Recommendations

Do feature engineering and hyperparameter optimization.
