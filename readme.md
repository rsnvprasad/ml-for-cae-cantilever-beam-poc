# ML for CAE: Cantilever Beam Prediction Explorer

Physics-guided Machine Learning for CAE using a cantilever beam problem.

This project demonstrates how Machine Learning can support engineering design exploration by comparing conventional physics calculations with selected ML models and interpolation approaches.

## Live App

🚀 Streamlit App:

https://ml-for-cae-cantilever-beam-poc.streamlit.app/

## Project Highlights

- Physics-based cantilever beam calculations
- DOE using ULH/LHS sampling
- Feature Engineering based on beam mechanics
- Multiple ML approaches:
  - Linear Regression
  - Random Forest
  - XGBoost
  - Neural Networks
  - RBF Interpolator

- Engineering validation framework:
  - Train/Validation/Test split
  - Cross-validation
  - Fair validation cases
  - Edge case validation

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- TensorFlow
- XGBoost
- Plotly
- Pandas

## Project Structure

```text
ml-for-cae-cantilever-beam-poc/
│
├── app/
├── app_data/
├── app_models/
├── config/
├── outputs/
├── src/
├── main.py
├── requirements.txt
└── readme.md
```

## Disclaimer

This project does not replace conventional CAE or engineering validation.

The objective is to accelerate design exploration and support engineering decision-making.
