import streamlit as st
import pandas as pd

baseline_df = pd.read_csv("baseline_bootstrap_predictions.csv")
m3_df = pd.read_csv("baseline_m3_bootstrap_predictions.csv")

st.title("Probability of an Unfavourable Outcome at 90-days")
st.write("Unfavourable outcome defined as mRS >2 at 90-days")
st.write("Repository for model development available at: https://github.com/PierogiCode/PCF.git")

model_choice = st.toggle("Use 24-hour model (M3)")

age = st.number_input("Age", min_value=30, max_value=90, value=65)
admission_nihss = st.number_input("Admission NIHSS", min_value=0, max_value=42, value=10)
ivt_label = st.selectbox("IVT", ["IVT not given", "IVT given"])
ivt = 0 if ivt_label == "IVT not given" else 1

sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_label == "Female" else 1

if model_choice:
    day_nihss = st.number_input("24-hour NIHSS", min_value=0, max_value=42, value=10)
    df = m3_df
else:
    df = baseline_df

if model_choice:
    row = df[
        (df.age == age) &
        (df.admission_NIHSS == admission_nihss) &
        (df.IVT == ivt) &
        (df.sex == sex) &
        (df.day_NIHSS == day_nihss)
    ]
else:
    row = df[
        (df.age == age) &
        (df.admission_NIHSS == admission_nihss) &
        (df.IVT == ivt) &
        (df.sex == sex)
    ]

if not row.empty:

    pred = row["pred_original"].values[0] * 100
    lower = row["pred_lower"].values[0] * 100
    upper = row["pred_upper"].values[0] * 100
    median = row["pred_median"].values[0] * 100

    st.metric("Predicted risk", f"{pred:.1f}%")
    st.caption(f"95% CI: {lower:.1f}% – {upper:.1f}% | Bootstrap median: {median:.1f}%")

    baseline_prob = row["baseline_probability"].values[0]

    age_contrib = (row["prob_age"].values[0] - baseline_prob) * 100
    nihss_contrib = (row["prob_admission_NIHSS"].values[0] - baseline_prob) * 100
    ivt_contrib = (row["prob_IVT"].values[0] - baseline_prob) * 100
    sex_contrib = (row["prob_sex"].values[0] - baseline_prob) * 100

    st.subheader("Feature Contributions to this Prediction:")
    st.write("*Derived from SHAP values*")

    st.write(f"Age: {age_contrib:+.1f}%")
    st.write(f"Admission NIHSS: {nihss_contrib:+.1f}%")
    st.write(f"IVT: {ivt_contrib:+.1f}%")
    st.write(f"Sex: {sex_contrib:+.1f}%")

    if model_choice:
        day_contrib = (row["prob_day_NIHSS"].values[0] - baseline_prob) * 100
        st.write(f"24-hour NIHSS: {day_contrib:+.1f}%")
