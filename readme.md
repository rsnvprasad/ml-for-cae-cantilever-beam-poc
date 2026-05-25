# ML for CAE: Cantilever Beam Prediction Explorer

A physics-guided Machine Learning proof of concept for CAE design-space exploration using a cantilever beam benchmark problem.

This project demonstrates how ML can support engineering understanding, faster design exploration, and early decision-making without replacing CAE or physics-based validation.

## Objective

The objective of this project is to understand how Machine Learning can be applied to engineering design and development workflows by comparing:

- Physics-based ground truth calculations
- Machine Learning regressors
- Neural Networks
- RBF-based surrogate / interpolation / Response Surface Models

## Problem Definition

The study uses a cantilever beam with:

- Length `L`
- Width `b`
- Height `h`
- Elastic modulus `E`
- Applied point load `F`

The predicted responses are:

- Deflection
- Bending stress

## Methodology

The project follows a complete ML-for-CAE workflow:

1. Data generation using ULH / LHS sampling
2. Physics-based response calculation
3. Sensitivity study using Pearson and Spearman correlations
4. Physics-guided feature engineering
5. Train / validation / test split
6. Model development and comparison
7. Cross-validation and iteration studies
8. Fair and edge-case engineering validation
9. Interactive Streamlit application

## Models Explored

The study started with Linear Regression as a baseline and progressed through:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks
- RBF Interpolators / Response Surface Models

The deployed app compares selected representative models:

- Random Forest
- XGBoost
- Neural Network
- RBF Interpolator

## App Features

The Streamlit app includes:

- Physics vs ML prediction comparison
- Best model identification for deflection and stress
- Engineering interpretation for each prediction
- Input range check against trained design space
- Model credibility panel
- R², MAE, and RMSE visualization
- Hyperparameter summary
- Fair and edge-case validation
- Data generation and sensitivity study visuals
- Feature engineering explanation

## Engineering Philosophy

This project is not intended to replace CAE or conventional engineering validation.

The intent is to explore how ML can support:

- Faster design-space exploration
- Early-stage engineering guidance
- Sensitivity understanding
- Quick comparison of candidate designs
- Better preparation before detailed CAE validation

Final engineering decisions should still rely on physics understanding, CAE validation, and conventional engineering review.

## Repository Structure

```text
app/
    app.py

app_data/
    Beam.png
    model_metrics_summary.csv

app_models/
    neural_network/

config/
    config.yml

outputs/
    plots/
        metrics_from_data_generation/
        sensitivity_analysis/

src/
    feature_builder.py
    model_loader.py
    physics.py
    predictor.py
    validation.py

main.py
requirements.txt