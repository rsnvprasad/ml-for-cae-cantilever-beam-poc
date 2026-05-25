from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predictor import predict_single_case
from src.validation import check_input_ranges, is_design_inside_training_range


st.set_page_config(
    page_title="Cantilever Beam Prediction Explorer",
    layout="wide",
)

st.title("Cantilever Beam Physics + ML Prediction Explorer")

st.markdown(
    """
    This app demonstrates how Machine Learning can support engineering design exploration
    by comparing physics-based cantilever beam calculations with selected ML and interpolation models.

    **Important:** This project is not intended to replace CAE or conventional engineering validation.
    It is intended to support faster exploration, early decision-making, and design-space understanding.
    """
)


left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.image("app_data/Beam.png", use_container_width=True)
    st.caption("Cantilever beam geometry and loading setup used for the ML study")

with right_col:
    with st.container(border=True):
        st.markdown("#### Choose Input Parameters for prediction")
        L_mm = st.number_input("Length - L(mm)", value=620.0)
        b_mm = st.number_input("Width - b(mm)", value=54.0)
        h_mm = st.number_input("Height - h(mm)", value=24.0)
        E_MPa = st.number_input("Elastic Modulus - E(MPa)", value=165000.0)
        F_N = st.number_input("Force - F(N)", value=1100.0)

        predict_clicked = st.button("Predict", use_container_width=True)

tab_predict, tab_credibility, tab_methodology = st.tabs(
    ["Prediction Explorer", "Model Credibility", "Methodology"]
)


def create_metrics_chart(metrics_data: pd.DataFrame, title: str):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=metrics_data["Model"],
            y=metrics_data["R2"],
            name="R²",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_data["Model"],
            y=metrics_data["MAE"],
            name="MAE",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_data["Model"],
            y=metrics_data["RMSE"],
            name="RMSE",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        legend_title="Metric",
        height=350,
    )

    fig.update_yaxes(title_text="R²", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="MAE / RMSE", secondary_y=True)

    return fig


if predict_clicked:
    range_results = check_input_ranges(L_mm, b_mm, h_mm, E_MPa, F_N)

    results_df = predict_single_case(
        L_mm=L_mm,
        b_mm=b_mm,
        h_mm=h_mm,
        E_MPa=E_MPa,
        F_N=F_N,
    )

    results_display = results_df.rename(
        columns={
            "Deflection Abs Error": "Deflection Error (mm)",
            "Deflection % Error": "Deflection Error (%)",
            "Stress Abs Error": "Stress Error (MPa)",
            "Stress % Error": "Stress Error (%)",
        }
    )

    ml_models = results_display[results_display["Model"] != "Ground Truth"]

    best_deflection = ml_models.loc[
        ml_models["Deflection Error (%)"].idxmin(),
        "Model",
    ]

    best_stress = ml_models.loc[
        ml_models["Stress Error (%)"].idxmin(),
        "Model",
    ]

    with tab_predict:
        if is_design_inside_training_range(range_results):
            st.success("Input values are within the trained design space.")
        else:
            st.warning(
                "One or more inputs are outside the trained design space. Predictions may be less reliable."
            )

        st.subheader("Predictions Comparison")

        st.success(
            f"🏆 Best Deflection Model: {best_deflection} "
            f"({ml_models['Deflection Error (%)'].min():.2f}% error)"
        )

        st.success(
            f"🏆 Best Stress Model: {best_stress} "
            f"({ml_models['Stress Error (%)'].min():.2f}% error)"
        )

        st.dataframe(
            results_display.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Engineering Interpretation")

        deflection_error = ml_models["Deflection Error (%)"].min()
        stress_error = ml_models["Stress Error (%)"].min()

        deflection_family = (
            "interpolation-based"
            if "RBF" in best_deflection
            else "regression-based"
        )

        stress_family = (
            "interpolation-based"
            if "RBF" in best_stress
            else "regression-based"
        )

        st.info(
            f"""
            For the current input combination, **{best_deflection}** gives the closest
            deflection prediction with **{deflection_error:.2f}% error**.

            For stress, **{best_stress}** gives the closest prediction with
            **{stress_error:.2f}% error**.

            This suggests that, for this specific selected design, the deflection response is
            better represented by a **{deflection_family}** approach, while the stress
            response is better represented by a **{stress_family}** approach.

            These observations are intended to support engineering judgment and faster
            design-space understanding. Final design decisions should still rely on
            physics understanding and conventional engineering review.
            """
        )

        st.info("""
        Model family interpretation:

        • Regression models:
        Random Forest (RF), XGBoost (XGB), Neural Networks (NN)

        Purpose:
        Learn global relationships and general trends across the design space.


        • Interpolation / Surrogate Models:
        RBF Interpolators (one type of surrogate model)

        Examples of surrogate methods:
        - RBF Interpolators
        - Response Surface Models (RSM)
        - Kriging models

        Purpose:
        Estimate responses from known design points while preserving local response behavior.


        Engineering interpretation:

        For some input combinations, regression models may generalize better, while interpolation models can perform strongly inside familiar regions of the trained design space.

        The intention is not to identify one universally best method, but to understand which modeling philosophy better represents the engineering response.
        """)

        st.subheader("Inputs Range Check")

        range_table = []
        for key, item in range_results.items():
            range_table.append(
                {
                    "Parameter": key,
                    "Input Value": item["value"],
                    "Training Min": item["min"],
                    "Training Max": item["max"],
                    "Inside Range": item["inside_range"],
                }
            )

        st.dataframe(
            range_table,
            use_container_width=True,
            hide_index=True,
        )

    with tab_credibility:
        st.subheader("Model Credibility")

        st.markdown(
            """
            Model trust should not come only from one prediction. It should also consider
            how each model performed on validation, unseen test data, and additional engineering 
            validation studies.

            In this study, the journey started with **Linear Regression** as a baseline,
            then progressed through **Random Forest, Gradient Boosting, XGBoost,
            Neural Networks, and RBF Interpolators**.

            For the interactive app comparison, selected representative models are shown:
            **Random Forest, XGBoost, Neural Network, and RBF Interpolator**.
            """
        )

        metrics_path = PROJECT_ROOT / "app_data" / "model_metrics_summary.csv"

        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)

            st.markdown("### Model Performance Visualization")

            deflection_metrics = metrics_df[metrics_df["Target"] == "Deflection"].copy()
            stress_metrics = metrics_df[metrics_df["Target"] == "Stress"].copy()

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_metrics_chart(deflection_metrics, "Deflection Metrics"),
                    use_container_width=True,
                )

            with col2:
                st.plotly_chart(
                    create_metrics_chart(stress_metrics, "Stress Metrics"),
                    use_container_width=True,
                )

            st.caption(
                "R² is shown on the left axis. MAE and RMSE are shown on the right axis. "
                "Higher R² and lower MAE/RMSE indicate better model performance."
            )

            with st.expander("Metrics Interpretation Guide"):

                st.markdown("""
            ### R² (Coefficient of Determination)
            st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")
            st.markdown("""
            Where:

            - `yᵢ` = actual value  
            - `ŷᵢ` = predicted value  
            - `ȳ` = mean of actual values  
            - `n` = number of samples
            """)

            Measures how well the model explains the response variation.

            **Interpretation:**
            - R² = 1.0 → Perfect prediction
            - R² > 0.95 → Excellent
            - R² ≈ 0.8–0.95 → Good
            - R² < 0.7 → Weak for engineering applications

            Engineering goal:
            - Maximize R² (closer to 1)

            ---

            ### MAE (Mean Absolute Error)
            st.latex(r"MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|")

            Average magnitude of prediction error.

            Example:

            If MAE = 20 MPa for stress,

            the prediction differs from the actual value by about 20 MPa on average.

            Engineering goal:
            - Lower is better
            - Easier to interpret physically

            ---

            ### RMSE (Root Mean Square Error)
            st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}")
            Similar to MAE, but penalizes larger prediction errors more strongly.

            Engineering goal:
            - Lower is better
            - Useful for detecting large prediction deviations

            ---

            ### General Engineering Interpretation

            Ideal model behavior:

            -> High R²  
            -> Low MAE  
            -> Low RMSE  

            However, model selection should also consider:

            - Generalization performance
            - Fair/Edge validation behavior
            - Physical consistency
            - Engineering plausibility
            """)
                
            with st.expander("View Neural Network Training Behavior"):

                st.markdown("""
                ### Neural Network Training Behavior

                Unlike tree-based models, Neural Networks learn through iterative optimization.
                During training, model performance is monitored using loss functions and learning-rate behavior.

                ---

                ### Loss Function

                The loss function measures how far predictions are from actual values.

                Typical behavior:

                • Higher loss → predictions are farther from actual values  
                • Lower loss → predictions are closer to actual values  
                • Goal → minimize loss during training

                Common loss functions:

                • Mean Squared Error (MSE)  
                • Mean Absolute Error (MAE)

                Engineering interpretation:

                • Continuous reduction in loss indicates learning progress  
                • Large fluctuations can indicate unstable learning  
                • Nearly constant loss may indicate underfitting

                ---

                ### Train vs Validation Loss

                Training loss -> Performance on training data

                Validation loss -> Performance on unseen validation data

                Typical interpretation:

                ✔ Train loss ↓ and Validation loss ↓
                → Good learning and generalization

                ⚠ Train loss ↓ and Validation loss ↑
                → Possible overfitting

                ⚠ Both losses remain high
                → Possible underfitting

                ---

                ### Learning Rate (LR) Convergence

                Learning rate controls how large each optimization step becomes.

                Engineering interpretation:

                • Very high learning rate:
                - Faster learning
                - Can overshoot optimum solution

                • Very low learning rate:
                - Stable learning
                - Slow convergence

                • Adaptive learning rate:
                - Starts larger and gradually reduces
                - Often improves convergence stability

                In this project:

                • Adam optimizer was used
                • Early stopping monitored validation behavior
                • Learning-rate reduction helped stabilize convergence
                """)

            with st.expander("View Model Hyperparameters"):
                hyperparameter_data = [
                    {
                        "Model": "Random Forest",
                        "Key Hyperparameters": (
                            "n_estimators=300, max_depth=12, min_samples_split=4, "
                            "min_samples_leaf=2, max_features=sqrt, random_state=42"
                        ),
                    },
                    {
                        "Model": "XGBoost",
                        "Key Hyperparameters": (
                            "n_estimators=400, learning_rate=0.05, max_depth=4, "
                            "subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0"
                        ),
                    },
                    {
                        "Model": "Neural Network",
                        "Key Hyperparameters": (
                            "architecture=17→6→4→1, activation=relu, optimizer=Adam, "
                            "learning_rate=0.0005, batch_size=32, max_epochs=500, "
                            "early_stopping_patience=50, l2=0.0001"
                        ),
                    },
                    {
                        "Model": "RBF Interpolator",
                        "Key Hyperparameters": (
                            "kernel=cubic, smoothing=0.0, neighbors=None, "
                            "feature_scaling=StandardScaler, target_scaling=None"
                        ),
                    },
                ]

                st.dataframe(
                    hyperparameter_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    """
                    Note:

                    The hyperparameters shown here are selected representative configurations and 
                    are not necessarily globally optimized parameters.

                    Multiple tuning iterations and model variants were explored during the study.
                    Additional tuning was intentionally stopped once model performance reached 
                    acceptable engineering levels and performance improvements became marginal.

                    The purpose of this project was not to maximize benchmark scores, but to 
                    understand how different ML approaches behave and can support engineering 
                    design exploration and decision-making.
                    """
                )

            with st.expander("View Detailed Metrics Table"):
                st.dataframe(
                    metrics_df.round(6),
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    """
                    The test metrics shown here are calculated on the final unseen test split, which represents 20% of the generated dataset. The remaining data was used for training and validation depending on the model family.\n
                    These metrics represent historical model performance on validation/test data.
                    They help build model credibility, but they do not guarantee accuracy for every new input case.
                    """
                )

            with st.expander("View Fair & Edge Engineering Validation"):
                st.markdown(
                    """
                    This section will summarize additional engineering validation cases.

                    These cases are separate from the 20% unseen test split.

                    - **Fair cases** represent realistic engineering design points.
                    - **Edge cases** represent challenging boundary or extreme-but-valid cases.
                    - The purpose is to understand model behavior beyond random test data.
                    """
                )

                st.markdown("### Validation Design Selection Context")

                validation_context = pd.DataFrame(
                    {
                        "Parameter": ["L (mm)", "b (mm)", "h (mm)", "E (MPa)", "F (N)"],
                        "Training Range": [
                            "300 – 1000",
                            "30 – 70",
                            "10 – 100",
                            "70,000 – 210,000",
                            "500 – 2000",
                        ],
                        "Fair Case Range": [
                            "460 – 780",
                            "42 – 63",
                            "19 – 31",
                            "90,000 – 185,000",
                            "900 – 1550",
                        ],
                        "Edge Case Range": [
                            "300 – 1000",
                            "30 – 60",
                            "40 – 100",
                            "70,000 – 210,000",
                            "800 – 2000",
                        ],
                    }
                )

                st.dataframe(
                    validation_context,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    """
                    Fair and Edge validation cases were selected from the broader 1000-design DOE space.

                    Fair cases represent balanced engineering design points.

                    Edge cases represent boundary or extreme-but-valid cases used to stress-test model robustness.
                    """
                )

                st.markdown("### Fair vs Edge Validation Summary")

                summary_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Average Deflection Error (%)",
                            "Average Stress Error (%)",
                            "Validation Purpose"
                        ],

                        "Fair Cases": [
                            "1.22 %",
                            "0.46 %",
                            "Representative engineering cases"
                        ],

                        "Edge Cases": [
                            "947.81 % †",
                            "171.32 % †",
                            "Extreme / robustness study"
                        ]
                    }
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    """
                    † Values are influenced by edge cases containing extremely small
                    physical response values. In such cases percentage error can appear
                    disproportionately large and should be interpreted together with
                    absolute error.
                    """
                )

                st.info("""
                Engineering note:

                For some edge cases with extremely small physical responses,
                percentage errors can become artificially large because the
                denominator (actual value) becomes very small.

                Therefore both absolute error and percentage error should be
                interpreted together.

                Extremely high percentage errors do not necessarily indicate
                catastrophic prediction failures.
                """)

                with st.expander("View Detailed Validation Cases"):
                    fair_cases = pd.DataFrame(
                        [
                            ["Fair", "V1", "Balanced mid-range steel-like case", 620, 54, 24, 165000, 1100, 62208, 8.51, 131.56, "XGBoost", 0.23, 2.73, "RBF Interpolator", 0.31, 0.24],
                            ["Fair", "V2", "Longer span moderate section stiffer material", 780, 48, 27, 185000, 1250, 78732, 13.58, 167.18, "Neural Network", 0.13, 0.94, "RBF Interpolator", 2.83, 1.69],
                            ["Fair", "V3", "Moderate span softer material higher load", 690, 58, 21, 115000, 1300, 44762, 27.65, 210.42, "RBF Interpolator", 0.28, 1.02, "RBF Interpolator", 0.07, 0.03],
                            ["Fair", "V4", "Shorter span slimmer width balanced load", 460, 42, 19, 150000, 900, 24007, 8.11, 163.83, "RBF Interpolator", 0.03, 0.39, "RBF Interpolator", 0.41, 0.25],
                            ["Fair", "V5", "Mid-long softer-material case with", 740, 63, 31, 90000, 1350, 156403, 12.95, 99.00, "Random Forest", 0.04, 0.32, "RBF Interpolator", 0.40, 0.40],
                            ["Fair", "V6", "Balanced mixed case for generalization", 560, 46, 26, 130000, 1550, 67375, 10.36, 167.48, "Linear Regression", 0.20, 1.93, "RBF TPS", 0.25, 0.15],
                        ],
                        columns=[
                            "Validation Type", "Case", "Description", "L (mm)", "b (mm)", "h (mm)",
                            "E (MPa)", "F (N)", "I (mm⁴)", "Actual Deflection (mm)",
                            "Actual Stress (MPa)", "Best Deflection Model", "Deflection Abs Error (mm)",
                            "Deflection Error (%)", "Best Stress Model", "Stress Abs Error (MPa)",
                            "Stress Error (%)",
                        ],
                    )

                    edge_cases = pd.DataFrame(
                        [
                            ["Edge", "C1", "Baseline Mid-Range Case", 500, 50, 80, 210000, 1000, 2133333, 0.09, 9.38, "Random Forest", 1.16, 1247.03, "Random Forest", 30.20, 322.19],
                            ["Edge", "C2", "High Deflection Case", 900, 40, 45, 70000, 1800, 303750, 20.57, 120.00, "XGBoost", 1.82, 8.83, "RF v3C", 0.42, 0.35],
                            ["Edge", "C3", "Low Deflection / Stiff Case", 300, 60, 100, 210000, 800, 5000000, 0.01, 2.40, "Random Forest", 0.23, 3374.47, "XGBoost", 12.36, 515.15],
                            ["Edge", "C4", "High Stress Case", 850, 35, 40, 210000, 2000, 186667, 10.44, 182.14, "Random Forest", 3.00, 28.76, "RBF Interpolator", 2.39, 1.31],
                            ["Edge", "C5", "Edge but Valid Mixed Case", 1000, 30, 55, 70000, 1500, 415938, 17.17, 99.17, "Random Forest", 13.74, 79.99, "RF v2", 17.45, 17.60],
                        ],
                        columns=fair_cases.columns,
                    )

                    st.markdown("#### Fair Validation Cases")
                    st.dataframe(
                        fair_cases,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("#### Edge Validation Cases")
                    st.dataframe(
                        edge_cases,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("#### Fair vs Edge Average Error Comparison")

                    validation_summary_chart = pd.DataFrame(
                        {
                            "Validation Type": ["Fair", "Fair", "Edge", "Edge"],
                            "Response": [
                                "Deflection Error (%)",
                                "Stress Error (%)",
                                "Deflection Error (%)",
                                "Stress Error (%)",
                            ],
                            "Average Error (%)": [
                                1.22,
                                0.46,
                                947.81,
                                171.32,
                            ],
                        }
                    )

                    fig_validation = go.Figure()

                    for response in validation_summary_chart["Response"].unique():
                        sub_df = validation_summary_chart[
                            validation_summary_chart["Response"] == response
                        ]

                        fig_validation.add_trace(
                            go.Bar(
                                x=sub_df["Validation Type"],
                                y=sub_df["Average Error (%)"],
                                name=response,
                            )
                        )

                    fig_validation.update_layout(
                        title="Fair vs Edge Validation: Average Percentage Error",
                        xaxis_title="Validation Type",
                        yaxis_title="Average Error (%) - Log Scale",
                        barmode="group",
                        height=420,
                    )

                    fig_validation.update_yaxes(type="log")

                    st.plotly_chart(
                        fig_validation,
                        use_container_width=True,
                    )

                    st.caption(
                        "Log scale is used because edge-case percentage errors are much larger than fair-case errors. "
                        "This helps compare both validation groups in the same chart. "
                        #"Use this chart together with detailed absolute error values for complete interpretation."
                    )

        else:
            st.warning(
                "Model metrics summary file not found. Please add it under app_data/model_metrics_summary.csv"
            )

    with tab_methodology:
        st.subheader("Project Methodology & Technical Background")

        st.markdown(
            """
            This project was built to understand how ML can be applied to engineering
            design and development workflows.

            The intent is **not to replace CAE or physics-based validation**.
            Instead, ML is explored as a support layer for:

            - Faster design-space exploration
            - Early-stage engineering guidance
            - Sensitivity understanding
            - Quick comparison of candidate designs
            - Better preparation before detailed CAE validation
            """
        )

        with st.expander("Data Generation using ULH / LHS Sampling"):

            st.success(
            """
            Dataset Size: 1000 samples

            Sampling Method -> Uniform Latin Hypercube (ULH/LHS)

            Purpose -> Broad design-space exploration
            """
            )

            st.markdown("### Design Space Coverage & Response Distributions")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_L_mm.png",
                    caption="Length Distribution",
                    use_container_width=True,
                )

                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_b_mm.png",
                    caption="Width Distribution",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_h_mm.png",
                    caption="Height Distribution",
                    use_container_width=True,
                )

                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_E_MPa.png",
                    caption="Elastic Modulus Distribution",
                    use_container_width=True,
                )

            with col3:
                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_I_mm4.png",
                    caption="Moment of Inertia (I) Distribution",
                    use_container_width=True,
                )

                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_F_N.png",
                    caption="Force Distribution",
                    use_container_width=True,
                )

            with col4:
                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_deflection_mm.png",
                    caption="Deflection Response Distribution",
                    use_container_width=True,
                )

                st.image(
                    "outputs/plots/metrics_from_data_generation/hist_stress_MPa.png",
                    caption="Stress Response Distribution",
                    use_container_width=True,
                )

            st.info(
                """
                Engineering observations:

                • ULH/LHS sampling provides broad coverage across the defined design space.

                • Input variables show near-uniform distributions, reducing sampling bias.

                • Response variables (deflection and stress) naturally develop non-uniform distributions due to underlying beam physics.

                • Wider response variation improves learning opportunities for nonlinear ML models.
                """
            )

        with st.expander("Sensitivity Study"):
            st.markdown("""
            Sensitivity analysis was performed to understand how input variables
            influence beam response and guide feature engineering.

            Two correlation methods were used:

            • Pearson → linear relationships

            • Spearman → monotonic relationships

            Purpose:

            • Identify strong/weak relationships
            • Validate physics trends
            • Support feature engineering
            • Improve model learning capability
            """)

            st.markdown("### Correlation Heatmaps")

            col1,col2 = st.columns(2)

            with col1:

                st.image(
                    "outputs/plots/sensitivity_analysis/all_features_pearson_heatmap.png",
                    caption="All Features — Pearson Correlation",
                    use_container_width=True
                )

            with col2:

                st.image(
                    "outputs/plots/sensitivity_analysis/all_features_spearman_heatmap.png",
                    caption="All Features — Spearman Correlation",
                    use_container_width=True
                )

            st.markdown("---")

            st.markdown("### Key Physics Relationships")

            col1,col2=st.columns(2)

            with col1:

                st.image(
                    "outputs/plots/sensitivity_analysis/transformed_L_mm_cubed_vs_deflection_mm.png",
                    caption="L³ vs Deflection",
                    use_container_width=True
                )

                st.image(
                    "outputs/plots/sensitivity_analysis/transformed_inv_I_mm4_vs_deflection_mm.png",
                    caption="1/I vs Deflection",
                    use_container_width=True
                )

            with col2:

                st.image(
                    "outputs/plots/sensitivity_analysis/transformed_inv_h_mm_sq_vs_stress_MPa.png",
                    caption="1/h² vs Stress",
                    use_container_width=True
                )

                st.image(
                    "outputs/plots/sensitivity_analysis/F_N_vs_stress_MPa.png",
                    caption="Force vs Stress",
                    use_container_width=True
                )


            st.info("""

            Engineering observations:

            • Deflection of the beam increases strongly with L³
            • Increase in beam stiffness reduces deflection
            • Stress shows strong inverse relation with h²
            • Force and Length of the beam contributes almost linearly to stress behavior
            • These trends aligned with beam theory and motivated feature engineering.

            """)

        with st.expander("Feature Engineering"):
            st.markdown(
                """
                Physics-based feature engineering was used to help ML models learn
                beam behavior more effectively.

                **Base inputs:**
                - `L_mm`
                - `b_mm`
                - `h_mm`
                - `E_MPa`
                - `F_N`

                **Engineered features:**
                - `L_cu`
                - `h_sq`
                - `h_cu`
                - `b_h2`
                - `b_h3`
                - `I_calc`
                - `inv_E`
                - `inv_I`
                - `F_L`
                - `F_L3`
                - `F_over_E`
                - `slenderness`

                These features were derived from cantilever beam deflection and
                bending stress relationships.
                """
            )

        with st.expander("Train / Validation / Test Split"):
            st.markdown(
                """
                The generated dataset contains **1000 design samples**.

                For model development, the data was split to evaluate model performance beyond
                the data used during training.

                **Primary split used:**

                - **Training data:** 80% of generated samples
                - **Test data:** 20% of generated samples

                The **20% test split** was kept unseen during model training and used to evaluate
                final model performance using R², MAE, and RMSE.

                **Validation / tuning approach:**

                Depending on the model family, validation was handled through validation split,
                tuning iterations, or cross-validation studies.

                **Purpose of each split:**

                - Training data: used to train the model
                - Validation data: used for tuning and monitoring
                - Test data: used for final unseen performance evaluation

                This avoids overconfidence and provides better visibility into generalization behavior.
                """
            )

        with st.expander("Model Training Strategy"):
            st.markdown(
                """
                The study followed a progressive model-development path:
                1. Linear Regression as baseline
                2. Random Forest
                3. Gradient Boosting
                4. XGBoost
                5. Neural Networks
                6. RBF Interpolators

                The app currently compares selected representative models:
                - Random Forest
                - XGBoost
                - Neural Network
                - RBF Interpolator

                The goal is not only to identify a best model, but to understand
                how different surrogate modeling methods behave for physics-driven
                engineering responses.
                """
            )

            st.markdown("""
            ### Why both Regressors and Interpolators?

            The study intentionally explored two different prediction philosophies:

            **Regression Models**\n
            (Linear Regression, Random Forest, Gradient Boosting, XGBoost, Neural Networks)

            Purpose:
            - Learn generalized relationships from data
            - Capture trends across the entire design space
            - Predict behavior for unseen combinations
            - Better suited for extrapolation/generalization studies

            Typical behavior:
            - Learn underlying patterns
            - May smooth local behavior
            - Performance depends on model complexity and training quality


            **Interpolation Models**\n
            (Radial Basis Function (RBF) Interpolators Models (RSM))

            Purpose:
            - Estimate outputs using neighboring known design points
            - Preserve local behavior within trained design space
            - Useful for rapid surrogate predictions

            Typical behavior:
            - Strong inside known design regions
            - Sensitive near boundaries or outside trained range
            - Less dependent on large model training effort


            Why compare both?

            The goal was not simply to identify one "best" algorithm. The comparison helps understand whether physics-driven engineering responses are better represented through:

            • global learned relationships (regression)

            or

            • local neighborhood behavior (interpolation)

            This provides deeper understanding for surrogate-model selection in future CAE workflows.
            """)

        with st.expander("Model Iterations & Cross-Validation Journey"):
            st.markdown(
                """
                The study was not limited to one model training attempt. Multiple
                iterations were explored to understand model behavior and improve
                confidence in the final comparison.

                **Random Forest journey:**
                - RF v1: initial baseline tree-based model
                - RF v2: tuned model with controlled depth and split parameters
                - RF v3 family: additional mini-sweep studies to compare behavior

                **Neural Network journey:**
                - NN v1: initial neural network model
                - NN v2: improved training setup and scaling approach
                - NN v3 Compact: reduced architecture for better generalization study

                **Cross-validation:**
                Cross-validation was also explored to understand whether model
                performance was stable across different data splits rather than being
                dependent on one random train/test split.

                **Final app selection:**

                The app shows selected representative models from the full study:
                - Random Forest
                - XGBoost
                - Neural Network
                - RBF Interpolator

                This keeps the interface compact while still representing the depth of
                the model-development journey.
                """
            )

else:
    with tab_predict:
        st.info("Enter input values in the sidebar and click Predict.")

    with tab_credibility:
        st.info("Run a prediction first to view model credibility details.")

    with tab_methodology:
        st.subheader("Project Methodology & Technical Background")
        st.markdown(
            """
            This project explores how Machine Learning can support engineering design
            exploration without replacing CAE or physics-based validation.

            Run a prediction to view the full methodology section along with model
            credibility and prediction results.
            """
        )