import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

st.set_page_config(page_title="Nepal Environmental AI Demo", page_icon="🌿", layout="wide")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

@st.cache_data
def generate_data(n=3000):
    provinces = ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"]
    eco_regions = ["Mountain", "Hill", "Terai"]
    seasons = ["Pre-monsoon", "Monsoon", "Post-monsoon", "Winter"]
    land_uses = ["Urban", "Agriculture", "Forest", "Riverbank", "Industrial"]
    municipality_types = ["Metropolitan", "Sub-metropolitan", "Municipality", "Rural Municipality"]
    fuels = ["LPG", "Firewood", "Electricity", "Biogas", "Kerosene"]
    waste_methods = ["Municipal collection", "Open dumping", "Burning", "River dumping", "Composting"]

    df = pd.DataFrame({
        "province": np.random.choice(provinces, n),
        "eco_region": np.random.choice(eco_regions, n, p=[0.25, 0.45, 0.30]),
        "season": np.random.choice(seasons, n),
        "land_use": np.random.choice(land_uses, n, p=[0.25, 0.25, 0.20, 0.15, 0.15]),
        "municipality_type": np.random.choice(municipality_types, n),
        "dominant_household_fuel": np.random.choice(fuels, n),
        "waste_disposal_method": np.random.choice(waste_methods, n),
        "elevation_m": np.random.gamma(2.0, 700, n),
        "slope_degree": np.random.gamma(2.0, 8, n),
        "annual_rainfall_mm": np.random.normal(1800, 550, n),
        "pm25_ugm3": np.random.normal(45, 18, n),
        "river_distance_km": np.random.exponential(3, n),
        "road_distance_km": np.random.exponential(4, n),
        "population_density_per_km2": np.random.gamma(2.5, 220, n),
        "forest_cover_percent": np.random.normal(42, 18, n),
        "waste_collection_coverage_percent": np.random.normal(55, 22, n),
        "industrial_activity_index": np.random.beta(2, 5, n) * 100,
        "ndvi_vegetation_index": np.random.beta(4, 3, n),
    })

    # Keep values realistic.
    df["annual_rainfall_mm"] = df["annual_rainfall_mm"].clip(400, 4500)
    df["pm25_ugm3"] = df["pm25_ugm3"].clip(5, 150)
    df["forest_cover_percent"] = df["forest_cover_percent"].clip(0, 100)
    df["waste_collection_coverage_percent"] = df["waste_collection_coverage_percent"].clip(0, 100)
    df["ndvi_vegetation_index"] = df["ndvi_vegetation_index"].clip(0, 1)

    # Rule-based synthetic risk signal for teaching.
    risk_score = (
        0.030 * df["pm25_ugm3"] +
        0.018 * df["slope_degree"] +
        0.0010 * df["annual_rainfall_mm"] +
        0.0012 * df["population_density_per_km2"] -
        0.018 * df["forest_cover_percent"] -
        0.012 * df["waste_collection_coverage_percent"] -
        1.3 * df["ndvi_vegetation_index"] +
        0.010 * df["industrial_activity_index"] +
        df["land_use"].map({"Industrial": 0.9, "Urban": 0.6, "Riverbank": 0.5, "Agriculture": 0.2, "Forest": -0.5}).fillna(0) +
        df["waste_disposal_method"].map({"Open dumping": 0.7, "Burning": 0.6, "River dumping": 0.8, "Municipal collection": -0.3, "Composting": -0.4}).fillna(0) +
        np.random.normal(0, 0.8, n)
    )
    threshold = np.quantile(risk_score, 0.58)
    df["high_environmental_risk"] = (risk_score > threshold).astype(int)

    # Add missing values to mimic real environmental data collection problems.
    for col in ["pm25_ugm3", "forest_cover_percent", "waste_collection_coverage_percent", "dominant_household_fuel"]:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan

    return df

@st.cache_resource
def train_models(df):
    X = df.drop(columns=["high_environmental_risk"])
    y = df["high_environmental_risk"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.08, eval_metric="logloss", random_state=RANDOM_STATE)
    if CAT_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(iterations=160, depth=4, learning_rate=0.08, verbose=False, random_state=RANDOM_STATE)

    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    best_name = results_df.iloc[0]["model"]
    return fitted, results_df, best_name, X_test, y_test


df = generate_data()
models, metrics, best_model_name, X_test, y_test = train_models(df)
best_model = models[best_model_name]


@st.cache_data
def get_feature_importance(_model, X_sample, y_sample):
    """Permutation importance explains which original input columns matter most globally."""
    perm = permutation_importance(
        _model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1"
    )
    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return importance


def make_default_input(df):
    """Create one complete model input row using safe median/mode defaults."""
    defaults = {}
    feature_df = df.drop(columns=["high_environmental_risk"])
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            defaults[col] = float(feature_df[col].median())
        else:
            defaults[col] = feature_df[col].mode(dropna=True).iloc[0]
    return defaults


def risk_driver_table(row_dict, importance_df):
    """Simple business-friendly local explanation based on user values + global importance."""
    med = df.drop(columns=["high_environmental_risk"]).median(numeric_only=True)
    records = []

    risk_maps = {
        "land_use": {"Industrial": "higher", "Urban": "higher", "Riverbank": "higher", "Agriculture": "medium", "Forest": "lower"},
        "waste_disposal_method": {"River dumping": "higher", "Open dumping": "higher", "Burning": "higher", "Municipal collection": "lower", "Composting": "lower"},
        "season": {"Monsoon": "higher", "Pre-monsoon": "medium", "Post-monsoon": "medium", "Winter": "lower"},
        "municipality_type": {"Metropolitan": "higher", "Sub-metropolitan": "medium", "Municipality": "medium", "Rural Municipality": "lower"},
    }

    numeric_direction = {
        "slope_degree": ("higher", "Steeper terrain can increase landslide/erosion vulnerability."),
        "annual_rainfall_mm": ("higher", "Heavy rainfall can increase flood, landslide, and runoff risk."),
        "pm25_ugm3": ("higher", "Higher PM2.5 indicates worse air-pollution exposure."),
        "forest_cover_percent": ("lower", "Lower forest cover reduces natural environmental protection."),
        "waste_collection_coverage_percent": ("lower", "Lower waste collection coverage increases unmanaged waste risk."),
        "industrial_activity_index": ("higher", "Higher industrial activity can increase pollution pressure."),
    }

    for _, r in importance_df.head(10).iterrows():
        f = r["feature"]
        v = row_dict.get(f, None)
        importance = float(r["importance_mean"])
        effect = "Neutral / context dependent"
        reason = "This feature is globally important for the trained model."

        if f in numeric_direction and f in med.index:
            direction, reason = numeric_direction[f]
            median_value = med[f]
            if direction == "higher":
                effect = "Pushes risk up" if float(v) > median_value else "Pushes risk down"
            else:
                effect = "Pushes risk up" if float(v) < median_value else "Pushes risk down"
        elif f in risk_maps:
            level = risk_maps[f].get(v, "medium")
            effect = "Pushes risk up" if level == "higher" else "Pushes risk down" if level == "lower" else "Medium influence"
            reason = f"Selected value '{v}' is treated as {level}-risk in the teaching signal."

        records.append({
            "feature": f,
            "your_value": v,
            "global_importance": importance,
            "local_effect": effect,
            "why_it_matters": reason,
        })
    return pd.DataFrame(records)


def build_recommendations(row_dict, probability):
    """Generate practical actions after prediction."""
    actions = []
    if probability >= 0.5:
        actions.append("Classify this case as priority for field verification and local monitoring.")
    else:
        actions.append("Keep this location in routine monitoring and update data regularly.")

    if row_dict["pm25_ugm3"] >= 55:
        actions.append("Air quality: investigate pollution sources, dust control, cleaner household fuel adoption, and industrial emission checks.")
    if row_dict["slope_degree"] >= 20 or row_dict["annual_rainfall_mm"] >= 2200:
        actions.append("Terrain/rainfall: prioritize landslide/flood early-warning checks, drainage review, and slope stabilization planning.")
    if row_dict["forest_cover_percent"] < 30:
        actions.append("Forest cover: consider reforestation, buffer-zone protection, and erosion-control measures.")
    if row_dict["waste_collection_coverage_percent"] < 50 or row_dict["waste_disposal_method"] in ["Open dumping", "River dumping", "Burning"]:
        actions.append("Waste management: improve collection coverage and reduce dumping/burning through municipal action plans.")
    if row_dict["industrial_activity_index"] >= 50 or row_dict["land_use"] == "Industrial":
        actions.append("Industrial pressure: review compliance, wastewater handling, and local pollution monitoring.")

    # Remove duplicates while preserving order.
    unique = []
    for a in actions:
        if a not in unique:
            unique.append(a)
    return unique

importance_df = get_feature_importance(best_model, X_test, y_test)

st.title("🌿 Nepal Environmental AI - Application")
st.caption("An intelligent demo platform for environmental risk prediction, visualization, and data-driven decision support in Nepal.")
st.caption("Example: data → preprocessing → model comparison → risk prediction → decision support")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dataset rows", f"{len(df):,}")
k2.metric("Features", f"{df.shape[1] - 1}")
k3.metric("Best model", best_model_name)
k4.metric("Best F1-score", f"{metrics.iloc[0]['f1_score']:.3f}")

st.markdown("""
### Demo objective
This app demonstrates how a simple user interface can be transformed into an interactive, useful AI product.
The model predicts whether a Nepal location-day is at High Environmental Risk by analyzing environmental, geographic, land-use, pollution, and infrastructure indicators. Based on the prediction result, the app also provides clear recommendations and next steps to support better decision-making.
""")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Data", "🤖 Model results", "🔮 Live prediction", "🧪 Top-10 risk check", "🧭 Guided top-10 decision", "🚀 Deployment concept"])

with tab1:
    st.subheader("Sample synthetic environmental dataset")
    st.dataframe(df.head(30), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        target_counts = df["high_environmental_risk"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Low/Moderate", "High"], target_counts.values)
        ax.set_title("Target Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        missing = df.isna().mean().sort_values(ascending=False).head(8) * 100
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(missing.index, missing.values)
        ax.set_title("Top Missing Value Columns")
        ax.set_xlabel("Missing %")
        st.pyplot(fig)

with tab2:
    st.subheader("Model comparison")
    st.dataframe(metrics.style.format({"accuracy":"{:.3f}", "precision":"{:.3f}", "recall":"{:.3f}", "f1_score":"{:.3f}", "roc_auc":"{:.3f}"}), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Final Test Metrics Comparison")
    ax.tick_params(axis='x', rotation=35)
    st.pyplot(fig)

    st.subheader("Top global feature importance")
    st.write("Permutation importance shows which original columns most affected the model's F1-score.")
    top_imp = importance_df.head(10).copy()
    st.dataframe(top_imp.style.format({"importance_mean":"{:.4f}", "importance_std":"{:.4f}"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_imp["feature"][::-1], top_imp["importance_mean"][::-1])
    ax.set_title(f"Top 10 Feature Importance - {best_model_name}")
    ax.set_xlabel("Permutation importance mean")
    st.pyplot(fig)

    pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    st.write(f"Confusion matrix for **{best_model_name}**")
    st.dataframe(pd.DataFrame(cm, index=["Actual Low/Moderate", "Actual High"], columns=["Predicted Low/Moderate", "Predicted High"]), use_container_width=True)

with tab3:
    st.subheader("Try a live risk prediction")
    st.write("Change the inputs below and see how the trained pipeline predicts environmental risk.")

    c1, c2, c3 = st.columns(3)
    with c1:
        province = st.selectbox("Province", sorted(df["province"].unique()), key="full_province")
        eco_region = st.selectbox("Eco-region", sorted(df["eco_region"].unique()), key="full_eco_region")
        season = st.selectbox("Season", sorted(df["season"].unique()), key="full_season")
        land_use = st.selectbox("Land use", sorted(df["land_use"].unique()), key="full_land_use")
        municipality_type = st.selectbox("Municipality type", sorted(df["municipality_type"].unique()), key="full_municipality_type")
        fuel = st.selectbox("Dominant household fuel", sorted(df["dominant_household_fuel"].dropna().unique()), key="full_fuel")
        waste = st.selectbox("Waste disposal method", sorted(df["waste_disposal_method"].unique()), key="full_waste")
    with c2:
        elevation = st.slider("Elevation (m)", 0, 5000, 1200, key="full_elevation")
        slope = st.slider("Slope degree", 0.0, 60.0, 12.0, key="full_slope")
        rainfall = st.slider("Annual rainfall (mm)", 400, 4500, 1800, key="full_rainfall")
        pm25 = st.slider("PM2.5 (ug/m3)", 5.0, 150.0, 45.0, key="full_pm25")
        river_distance = st.slider("River distance (km)", 0.0, 30.0, 3.0, key="full_river_distance")
        road_distance = st.slider("Road distance (km)", 0.0, 30.0, 4.0, key="full_road_distance")
    with c3:
        pop_density = st.slider("Population density/km²", 0, 3000, 600, key="full_population_density")
        forest_cover = st.slider("Forest cover %", 0.0, 100.0, 40.0, key="full_forest_cover")
        waste_collection = st.slider("Waste collection coverage %", 0.0, 100.0, 55.0, key="full_waste_collection")
        industrial = st.slider("Industrial activity index", 0.0, 100.0, 25.0, key="full_industrial_activity")
        ndvi = st.slider("NDVI vegetation index", 0.0, 1.0, 0.55, key="full_ndvi")

    row = pd.DataFrame([{
        "province": province,
        "eco_region": eco_region,
        "season": season,
        "land_use": land_use,
        "municipality_type": municipality_type,
        "dominant_household_fuel": fuel,
        "waste_disposal_method": waste,
        "elevation_m": elevation,
        "slope_degree": slope,
        "annual_rainfall_mm": rainfall,
        "pm25_ugm3": pm25,
        "river_distance_km": river_distance,
        "road_distance_km": road_distance,
        "population_density_per_km2": pop_density,
        "forest_cover_percent": forest_cover,
        "waste_collection_coverage_percent": waste_collection,
        "industrial_activity_index": industrial,
        "ndvi_vegetation_index": ndvi,
    }])

    probability = best_model.predict_proba(row)[0, 1]
    prediction = "High Environmental Risk" if probability >= 0.5 else "Low/Moderate Environmental Risk"
    st.metric("Predicted risk", prediction, f"High-risk probability: {probability:.1%}")
    if probability >= 0.5:
        st.warning("Recommended action: prioritize monitoring, site visit, pollution/waste review, and mitigation planning.")
    else:
        st.success("Recommended action: continue routine monitoring and update data regularly.")


with tab4:
    st.subheader("Top-10 feature risk check")
    st.write("Use only the most important features from the notebook/model comparison to predict risk as **0 or 1**, then view recommended actions and the main drivers.")

    top10_features = importance_df.head(10)["feature"].tolist()
    defaults = make_default_input(df)

    st.markdown("**Top 10 model features used in this simplified demo**")
    st.dataframe(importance_df.head(10).reset_index(drop=True).style.format({"importance_mean":"{:.4f}", "importance_std":"{:.4f}"}), use_container_width=True)

    a, b = st.columns(2)
    with a:
        slope_degree = st.slider("Slope degree", 0.0, 60.0, 18.0, help="Higher slope usually increases erosion/landslide risk.", key="top10_slope")
        land_use = st.selectbox("Land use", sorted(df["land_use"].dropna().unique()), index=sorted(df["land_use"].dropna().unique()).index("Urban"), key="top10_land_use")
        annual_rainfall_mm = st.slider("Annual rainfall (mm)", 400, 4500, 2200, key="top10_rainfall")
        pm25_ugm3 = st.slider("PM2.5 (ug/m3)", 5.0, 150.0, 60.0, key="top10_pm25")
        forest_cover_percent = st.slider("Forest cover %", 0.0, 100.0, 28.0, key="top10_forest_cover")
    with b:
        waste_collection_coverage_percent = st.slider("Waste collection coverage %", 0.0, 100.0, 45.0, key="top10_waste_collection")
        industrial_activity_index = st.slider("Industrial activity index", 0.0, 100.0, 45.0, key="top10_industrial_activity")
        waste_disposal_method = st.selectbox("Waste disposal method", sorted(df["waste_disposal_method"].dropna().unique()), key="top10_waste_disposal")
        season = st.selectbox("Season", sorted(df["season"].dropna().unique()), key="top10_season")
        municipality_type = st.selectbox("Municipality type", sorted(df["municipality_type"].dropna().unique()), key="top10_municipality_type")

    manual_values = {
        "slope_degree": slope_degree,
        "land_use": land_use,
        "annual_rainfall_mm": annual_rainfall_mm,
        "pm25_ugm3": pm25_ugm3,
        "forest_cover_percent": forest_cover_percent,
        "waste_collection_coverage_percent": waste_collection_coverage_percent,
        "industrial_activity_index": industrial_activity_index,
        "waste_disposal_method": waste_disposal_method,
        "season": season,
        "municipality_type": municipality_type,
    }

    full_input = defaults.copy()
    full_input.update(manual_values)
    input_row = pd.DataFrame([full_input])

    st.divider()
    if st.button("Predict risk from top-10 features", type="primary", key="top10_predict_button"):
        proba = float(best_model.predict_proba(input_row)[0, 1])
        pred_class = int(proba >= 0.5)

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted class", pred_class)
        r2.metric("Meaning", "High Risk" if pred_class == 1 else "Low/Moderate Risk")
        r3.metric("High-risk probability", f"{proba:.1%}")

        if pred_class == 1:
            st.error("Result: **1 = High Environmental Risk**. This case should be prioritized for review.")
        else:
            st.success("Result: **0 = Low/Moderate Environmental Risk**. Continue regular monitoring.")

        st.subheader("Recommended actions")
        for action in build_recommendations(full_input, proba):
            st.write("✅ " + action)

        st.subheader("Important features driving this prediction")
        driver_df = risk_driver_table(full_input, importance_df)
        st.dataframe(driver_df.style.format({"global_importance":"{:.4f}"}), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = driver_df.copy().sort_values("global_importance", ascending=True)
        ax.barh(plot_df["feature"], plot_df["global_importance"])
        ax.set_title("Top Feature Influence Used for Explanation")
        ax.set_xlabel("Global permutation importance")
        st.pyplot(fig)

        st.info("Teaching note: this explanation combines global permutation importance with simple local business rules. In real production, you could add SHAP/LIME for stronger local explanations.")

with tab5:
    st.subheader("Guided top-10 decision assistant")
    st.write(
        "This tab is designed for students and non-technical users. Instead of typing exact numbers, "
        "users select values from simple dropdown menus for the top risk-driving features. The app then "
        "predicts risk class 0 or 1, explains why, and gives next steps."
    )

    defaults_guided = make_default_input(df)

    st.markdown("### Step 1 — Enter the top 10 important feature values")
    g1, g2 = st.columns(2)

    with g1:
        slope_choice = st.selectbox(
            "Slope degree level",
            ["Low slope: 0-10 degrees", "Medium slope: 10-20 degrees", "High slope: 20-35 degrees", "Very high slope: 35-60 degrees"],
            index=1,
            key="guided_slope_choice",
        )
        rainfall_choice = st.selectbox(
            "Annual rainfall level",
            ["Low rainfall: 400-1,200 mm", "Medium rainfall: 1,200-2,000 mm", "High rainfall: 2,000-3,000 mm", "Very high rainfall: 3,000-4,500 mm"],
            index=2,
            key="guided_rainfall_choice",
        )
        pm25_choice = st.selectbox(
            "PM2.5 air pollution level",
            ["Good/low: 5-25 ug/m3", "Moderate: 25-50 ug/m3", "Unhealthy: 50-90 ug/m3", "Very unhealthy: 90-150 ug/m3"],
            index=2,
            key="guided_pm25_choice",
        )
        forest_choice = st.selectbox(
            "Forest cover level",
            ["Very low forest cover: 0-20%", "Low forest cover: 20-40%", "Medium forest cover: 40-60%", "High forest cover: 60-100%"],
            index=1,
            key="guided_forest_choice",
        )
        waste_collection_choice = st.selectbox(
            "Waste collection coverage level",
            ["Poor coverage: 0-30%", "Limited coverage: 30-50%", "Moderate coverage: 50-75%", "Good coverage: 75-100%"],
            index=1,
            key="guided_waste_collection_choice",
        )

    with g2:
        industrial_choice = st.selectbox(
            "Industrial activity level",
            ["Low industrial activity: 0-20", "Medium industrial activity: 20-50", "High industrial activity: 50-75", "Very high industrial activity: 75-100"],
            index=1,
            key="guided_industrial_choice",
        )
        land_use_guided = st.selectbox("Land use", sorted(df["land_use"].dropna().unique()), key="guided_land_use")
        waste_method_guided = st.selectbox("Waste disposal method", sorted(df["waste_disposal_method"].dropna().unique()), key="guided_waste_method")
        season_guided = st.selectbox("Season", sorted(df["season"].dropna().unique()), key="guided_season")
        municipality_guided = st.selectbox("Municipality type", sorted(df["municipality_type"].dropna().unique()), key="guided_municipality")

    guided_numeric_map = {
        "Low slope: 0-10 degrees": 5.0,
        "Medium slope: 10-20 degrees": 15.0,
        "High slope: 20-35 degrees": 28.0,
        "Very high slope: 35-60 degrees": 45.0,
        "Low rainfall: 400-1,200 mm": 900.0,
        "Medium rainfall: 1,200-2,000 mm": 1600.0,
        "High rainfall: 2,000-3,000 mm": 2500.0,
        "Very high rainfall: 3,000-4,500 mm": 3700.0,
        "Good/low: 5-25 ug/m3": 18.0,
        "Moderate: 25-50 ug/m3": 38.0,
        "Unhealthy: 50-90 ug/m3": 70.0,
        "Very unhealthy: 90-150 ug/m3": 115.0,
        "Very low forest cover: 0-20%": 10.0,
        "Low forest cover: 20-40%": 30.0,
        "Medium forest cover: 40-60%": 50.0,
        "High forest cover: 60-100%": 75.0,
        "Poor coverage: 0-30%": 20.0,
        "Limited coverage: 30-50%": 40.0,
        "Moderate coverage: 50-75%": 62.0,
        "Good coverage: 75-100%": 85.0,
        "Low industrial activity: 0-20": 10.0,
        "Medium industrial activity: 20-50": 35.0,
        "High industrial activity: 50-75": 62.0,
        "Very high industrial activity: 75-100": 88.0,
    }

    guided_values = {
        "slope_degree": guided_numeric_map[slope_choice],
        "annual_rainfall_mm": guided_numeric_map[rainfall_choice],
        "pm25_ugm3": guided_numeric_map[pm25_choice],
        "forest_cover_percent": guided_numeric_map[forest_choice],
        "waste_collection_coverage_percent": guided_numeric_map[waste_collection_choice],
        "industrial_activity_index": guided_numeric_map[industrial_choice],
        "land_use": land_use_guided,
        "waste_disposal_method": waste_method_guided,
        "season": season_guided,
        "municipality_type": municipality_guided,
    }

    st.markdown("### Step 2 — Review the model input values")
    review_df = pd.DataFrame([{"Feature": k, "Selected value sent to model": v} for k, v in guided_values.items()])
    st.dataframe(review_df, use_container_width=True)

    guided_full_input = defaults_guided.copy()
    guided_full_input.update(guided_values)
    guided_row = pd.DataFrame([guided_full_input])

    if st.button("Predict and generate recommendations", type="primary", key="guided_predict_button"):
        guided_proba = float(best_model.predict_proba(guided_row)[0, 1])
        guided_pred = int(guided_proba >= 0.5)

        st.markdown("### Step 3 — Prediction result")
        p1, p2, p3 = st.columns(3)
        p1.metric("Prediction class", guided_pred)
        p2.metric("Meaning", "High Environmental Risk" if guided_pred == 1 else "Low/Moderate Environmental Risk")
        p3.metric("High-risk probability", f"{guided_proba:.1%}")

        if guided_pred == 1:
            st.error("The model predicts **1 = High Environmental Risk**. Treat this area as a priority case for validation and action.")
        else:
            st.success("The model predicts **0 = Low/Moderate Environmental Risk**. Continue monitoring, but this is not the highest-priority case based on current inputs.")

        st.markdown("### Step 4 — Important features behind this prediction")
        guided_driver_df = risk_driver_table(guided_full_input, importance_df)
        st.dataframe(guided_driver_df.style.format({"global_importance":"{:.4f}"}), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = guided_driver_df.copy().sort_values("global_importance", ascending=True)
        ax.barh(plot_df["feature"], plot_df["global_importance"])
        ax.set_title("Important Features Behind the Decision")
        ax.set_xlabel("Global permutation importance")
        st.pyplot(fig)

        st.markdown("### Step 5 — What needs to be done next")
        if guided_pred == 1:
            st.markdown("""
            **Immediate priority actions:**
            1. Validate the input data using field officers, local municipality records, or sensor readings.
            2. Mark the location for priority monitoring because multiple risk factors may be combining.
            3. Prepare a field checklist covering air pollution, waste disposal, drainage, river area, slope stability, and forest loss.
            4. Assign responsibility to the local environment office, ward office, disaster-risk team, or municipal waste team.
            5. Re-run the prediction after corrective action or after new data is collected.
            """)
        else:
            st.markdown("""
            **Routine monitoring actions:**
            1. Keep collecting updated air, rainfall, land-use, forest, and waste-management data.
            2. Watch the top driver features because low/moderate risk can become high risk if conditions worsen.
            3. Improve the weakest areas first, especially waste collection, forest cover, air quality, or slope/drainage controls.
            4. Recheck the prediction monthly, seasonally, or after major weather events.
            """)

        st.markdown("### Step 6 — Specific recommendation checklist")
        for action in build_recommendations(guided_full_input, guided_proba):
            st.write("✅ " + action)

        st.info(
            "Teaching note: this is a production-style decision-support demo. "
            "It helps prioritize review and resource planning, but final decisions should include expert field validation."
        )

with tab6:
    st.subheader("How this notebook becomes a demo product")
    st.markdown("""
    **Notebook stage:** experiment, explain, compare models.  
    **Demo app stage:** convert the trained workflow into an interactive interface.  
    **Production stage:** add real data pipelines, authentication, monitoring, scheduled retraining, logging, and governance.

    You do not need AWS, Azure, or GCP. You can run this app locally or publish it using a free demo hosting platform such as Streamlit Community Cloud. It works well for research, visualization, and simple application demos.
    """)
