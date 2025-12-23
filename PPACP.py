import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Campaign Orders Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers & Cached loaders
# -------------------------
@st.cache_resource
def load_pipeline(pipeline_path="final_pipeline_lgbm.pkl"):
    p = Path(pipeline_path)
    if not p.exists():
        raise FileNotFoundError(f"{pipeline_path} not found. Place final_pipeline_lgbm.pkl in the app folder.")
    return joblib.load(p)

@st.cache_data
def load_dataset_if_exists(path="products_campaign_cleaned.csv"):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None

def compute_defaults(pipeline, df=None):
    # If dataset present compute medians; otherwise fallback to sensible values
    features = pipeline.get("features", [])
    defaults = {}
    if df is not None:
        med = df[features].median()
        for f in features:
            # convert to simple python types to prevent Streamlit widget issues
            val = med.get(f, 0)
            defaults[f] = float(val) if np.isfinite(val) else 0.0
    else:
        # sensible fallbacks (based on earlier EDA)
        fallback = {
            "limit_infor": 0,
            "campaign_type": 3,
            "campaign_level": 1,
            "product_level": 1,
            "resource_amount": 5,
            "email_rate": 0.49,
            "price": 163,
            "discount_rate": 0.82,
            "hour_resouces": 713,
            "campaign_fee": 3662,
            "effective_price": 163 - (163*0.82),
            "resource_efficiency": 4563/(713+1),
            "total_campaign_cost": 3662 + (713 * 0.5)
        }
        for f in features:
            defaults[f] = float(fallback.get(f, 0.0))
    return defaults

def make_input_df(features, input_vals):
    df = pd.DataFrame([input_vals], columns=features)
    return df

def pretty_kpi(col1, col2, col3, col4):
    c1, c2, c3, c4 = st.columns([1.5,1,1,1])
    c1.metric(col1["label"], col1["value"], delta=col1.get("delta"))
    c2.metric(col2["label"], col2["value"], delta=col2.get("delta"))
    c3.metric(col3["label"], col3["value"], delta=col3.get("delta"))
    c4.metric(col4["label"], col4["value"], delta=col4.get("delta"))

# -------------------------
# Load model / pipeline
# -------------------------
st.title("🚀 Product Campaign Orders Predictor")
st.markdown(
    """
    **Predict expected orders for a product campaign.**  
    Use the controls on the left to change campaign settings and instantly see predicted orders, uncertainty, and diagnostics.
    """
)

# Attempt load pipeline
try:
    pipeline = load_pipeline("final_pipeline_lgbm.pkl")
    scaler = pipeline["scaler"]
    model = pipeline["model"]
    features = pipeline["features"]
except Exception as e:
    st.error(f"Could not load pipeline: {e}")
    st.stop()

# Try to load dataset for smart defaults
df = load_dataset_if_exists("products_campaign_cleaned.csv")
defaults = compute_defaults(pipeline, df)

# -------------------------
# Sidebar Inputs & Presets
# -------------------------
with st.sidebar:
    st.header("Campaign Inputs")
    st.markdown("Quick presets:")
    preset = st.selectbox("Choose a preset", options=["Median Campaign", "High-Budget", "Low-Budget", "Custom"])

    # Provide presets
    if preset == "Median Campaign":
        preset_vals = {k: defaults[k] for k in features}
        # small rounding for discrete fields
        preset_vals["campaign_type"] = int(round(preset_vals.get("campaign_type", 3)))
        preset_vals["campaign_level"] = int(round(preset_vals.get("campaign_level", 1)))
        preset_vals["product_level"] = int(round(preset_vals.get("product_level", 1)))
    elif preset == "High-Budget":
        preset_vals = {k: defaults[k] for k in features}
        preset_vals["campaign_fee"] = float(defaults.get("campaign_fee", 3662)) * 2.5
        preset_vals["hour_resouces"] = float(defaults.get("hour_resouces", 713)) * 1.5
        preset_vals["resource_amount"] = float(defaults.get("resource_amount", 5)) + 2
    elif preset == "Low-Budget":
        preset_vals = {k: defaults[k] for k in features}
        preset_vals["campaign_fee"] = float(defaults.get("campaign_fee", 3662)) * 0.4
        preset_vals["hour_resouces"] = float(defaults.get("hour_resouces", 713)) * 0.6
        preset_vals["resource_amount"] = max(1, float(defaults.get("resource_amount", 5)) - 2)
    else:
        preset_vals = {k: defaults[k] for k in features}

    # Create interactive inputs grouped logically
    st.subheader("Campaign & Product")
    limit_infor = st.number_input("limit_infor", value=int(round(preset_vals.get("limit_infor",0))), step=1)
    campaign_type = st.selectbox("campaign_type (channel)", options=list(range(0,7)), index=int(round(preset_vals.get("campaign_type",3))))
    campaign_level = st.selectbox("campaign_level (0=local,1=national)", options=[0,1], index=int(round(preset_vals.get("campaign_level",1))))
    product_level = st.selectbox("product_level (1=budget,2=mid,3=premium)", options=[1,2,3], index=int(round(preset_vals.get("product_level",1))))
    resource_amount = st.slider("resource_amount (units)", min_value=1, max_value=20, value=int(round(preset_vals.get("resource_amount",5))))
    
    st.subheader("Performance & Pricing")
    email_rate = st.slider("email_rate (open rate)", min_value=0.0, max_value=1.0, value=float(round(preset_vals.get("email_rate",0.49),2)), step=0.01)
    price = st.number_input("price", min_value=1.0, max_value=10000.0, value=float(round(preset_vals.get("price",163))), format="%.2f")
    discount_rate = st.slider("discount_rate (0-1)", min_value=0.0, max_value=1.0, value=float(round(preset_vals.get("discount_rate",0.82),2)), step=0.01)
    effective_price = st.number_input("effective_price (price - discount_value)", value=float(round(preset_vals.get("effective_price", price - price*discount_rate),2)))
    
    st.subheader("Operations & Cost")
    hour_resouces = st.number_input("hour_resouces (man-hours)", min_value=0.0, value=float(round(preset_vals.get("hour_resouces",713))), step=1.0)
    campaign_fee = st.number_input("campaign_fee (currency)", min_value=0.0, value=float(round(preset_vals.get("campaign_fee",3662))), step=1.0)

    # Derived features update button
    if st.button("Recompute derived features"):
        effective_price = price - (price * discount_rate)
        total_campaign_cost = campaign_fee + (hour_resouces * 0.5)
        resource_efficiency = 0 if hour_resouces == 0 else (defaults.get("effective_price", price) / (hour_resouces + 1))  # placeholder
        st.success("Derived features recomputed (you can fine tune values manually).")
    else:
        total_campaign_cost = campaign_fee + (hour_resouces * 0.5)
        resource_efficiency = 0 if hour_resouces == 0 else ( (defaults.get("resource_efficiency", 4563/(713+1))) )

    st.markdown("---")
    st.write("Model & action")
    show_shap = st.checkbox("Show SHAP (if available)", value=False)
    run_predict = st.button("Predict orders")

# -------------------------
# Main layout: KPIs, charts, prediction
# -------------------------
left, right = st.columns((2,1))

with left:
    st.subheader("Inputs summary")
    inputs = {
        "limit_infor": limit_infor,
        "campaign_type": campaign_type,
        "campaign_level": campaign_level,
        "product_level": product_level,
        "resource_amount": resource_amount,
        "email_rate": float(email_rate),
        "price": float(price),
        "discount_rate": float(discount_rate),
        "hour_resouces": float(hour_resouces),
        "campaign_fee": float(campaign_fee),
        "effective_price": float(effective_price),
        "resource_efficiency": float(resource_efficiency),
        "total_campaign_cost": float(total_campaign_cost)
    }
    inputs_df = make_input_df(features, inputs)
    st.dataframe(inputs_df.T, height=320)

    # Predict block
    if run_predict:
        # scale & predict
        X_new = inputs_df[features]
        X_new_scaled = scaler.transform(X_new)
        pred = model.predict(X_new_scaled)[0]
        pred_round = int(np.round(pred))

        # Quick uncertainty estimate: use small ensemble if raw model supports `predict` on several models saved
        # We'll check for other model files (rf/xgb) and compute simple std across their predictions if available.
        preds_for_uncert = [pred]
        # try to load alternative models if present
        alt_models = []
        for alt_name in ["model_rf_v1763026022.pkl", "model_xgb_v1763026022.pkl", "model_gbr_v1763026022.pkl"]:
            try:
                m = joblib.load(alt_name)
                alt_models.append(m)
            except:
                pass
        for m in alt_models:
            try:
                preds_for_uncert.append(m.predict(X_new_scaled)[0])
            except:
                pass
        pred_std = float(np.std(preds_for_uncert)) if len(preds_for_uncert) > 1 else 0.0

        # KPIs
        kpi1 = {"label":"Predicted Orders", "value": f"{pred_round}"}
        kpi2 = {"label":"Point estimate", "value": f"{pred:.1f}", "delta": f"{pred_std:.1f}"}
        kpi3 = {"label":"Effective Price", "value": f"{inputs['effective_price']:.2f}"}
        kpi4 = {"label":"Total Campaign Cost", "value": f"{inputs['total_campaign_cost']:.2f}"}
        pretty_kpi(kpi1,kpi2,kpi3,kpi4)

        st.markdown("### 📊 Prediction details")
        st.write(f"**Rounded prediction:** {pred_round} orders")
        st.write(f"**Raw prediction:** {pred:.2f}")
        st.write(f"**Uncertainty (approx. std from alternatives):** {pred_std:.2f}")

        # Downloadable result
        out = X_new.copy()
        out["predicted_orders"] = pred_round
        csv = out.to_csv(index=False)
        st.download_button("Download prediction as CSV", data=csv, file_name="campaign_prediction.csv")

        # Plot predicted orders vs some dataset distribution if available
        if df is not None:
            fig = px.histogram(df, x="orders", nbins=40, title="Historic distribution of orders")
            fig.add_vline(x=pred, line_color="red", annotation_text="Predicted", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        # Show residuals from test-set predictions if predictions CSV exists
        try:
            preds_df = pd.read_csv("final_test_predictions.csv")
            fig2 = px.scatter(preds_df, x="predicted_orders", y="residual", title="Residuals vs Predicted (Test set)")
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

        # Feature importance
        try:
            fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            fig3 = px.bar(fi.head(12), x="importance", y="feature", orientation="h", title="Top Feature Importances (LightGBM)")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            st.info("Feature importance not available for this model.")

        # SHAP (optional)
        if show_shap:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(scaler.transform(pd.DataFrame([inputs], columns=features)))
                st.subheader("SHAP summary (global on test set if available)")
                if df is not None:
                    # compute shap on a sample of df
                    sample = df[features].sample(min(200, len(df)), random_state=42)
                    sample_scaled = scaler.transform(sample)
                    expl = shap.TreeExplainer(model)
                    svals = expl.shap_values(sample_scaled)
                    shap.summary_plot(svals, features=sample, feature_names=features, show=False)
                    st.pyplot(bbox_inches="tight")
                else:
                    st.info("No dataset in folder to compute global SHAP; showing local explanation.")
                    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([inputs], columns=features), matplotlib=True, show=False)
                    st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"SHAP failed: {e}")

with right:
    st.subheader("Model Diagnostics & Notes")
    st.markdown(
        """
        - Model: **LightGBM (tuned)** (deployed pipeline: `final_pipeline_lgbm.pkl`)  
        - Make sure the pipeline file is placed in the same folder as this app.  
        - Use presets to quickly test scenarios (High-Budget, Low-Budget).  
        - SHAP is optional - install `shap` to enable per-feature explainability.
        """
    )
    st.markdown("### Quick tips")
    st.markdown(
        """
        - Increase `campaign_fee` and `hour_resouces` to see how predictions respond.  
        - Watch `resource_efficiency` — a high value often indicates good orders per hour.  
        - MAPE under ~10% is generally strong for campaign forecasts.
        """
    )

    st.markdown("---")
    st.markdown("### Files detected in working folder")
    files = [f for f in Path(".").iterdir() if f.is_file()]
    file_list = [f.name for f in files if any(ext in f.name.lower() for ext in [".pkl", ".csv"])]
    st.write(file_list)

# Footer
st.markdown("---")
st.caption("Built by Rachit Patwa")
