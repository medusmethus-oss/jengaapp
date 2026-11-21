import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(
    page_title="JengaApp — Nairobi Real Estate Development & Audit Portal",
    page_icon="🏢",
    layout="wide"
)

# Custom transformer definition required so joblib unpickles cleanly in production
class RealEstateLedgerFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        memo = df["memo_raw"].astype(str)
        feats = pd.DataFrame(index=df.index)
        feats["is_wip"] = (df["ledger_key"] == "wip_incurred").astype(int)
        feats["is_bill"] = (df["entry_type"] == "Bill").astype(int)
        feats["is_multiple"] = memo.str.contains("multiple", case=False, na=False).astype(int)
        feats["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
        feats["memo_len"] = memo.str.len()
        feats["has_qty_unit"] = memo.str.contains(r"\d+\s*(?:PCS|PS|BAGS|PAGS|LORRY|BOOKS|PAIR|KG|MM|X)", case=False, regex=True).astype(int)
        feats["has_cheque_or_inv"] = memo.str.contains(r"CHQ|INV|CHEQUE|#", case=False, regex=True).astype(int)
        feats["day_of_month"] = df["Date"].dt.day
        feats["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12.0)
        feats["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12.0)
        return feats

@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join("Zamil_Consultancy_Deliverables", "zamil_phase2_engine.joblib"))

bundle = load_bundle()

NAIROBI_ZONES = pd.DataFrame([
    {"Region": "Eastleigh / Pangani", "Lat": -1.2762, "Lon": 36.8482, "Land_Per_Acre_KES": 260_000_000, "Build_Cost_Per_Sqm_KES": 52_000, "Sale_Price_Per_Sqm_KES": 68_000, "Monthly_Rent_Per_Sqm_KES": 480, "Capital_Growth_%": 6.4, "Zoning_Note": "High-density commercial/residential; strong cash-buyer velocity"},
    {"Region": "South C / Nairobi West", "Lat": -1.3125, "Lon": 36.8256, "Land_Per_Acre_KES": 210_000_000, "Build_Cost_Per_Sqm_KES": 56_000, "Sale_Price_Per_Sqm_KES": 78_000, "Monthly_Rent_Per_Sqm_KES": 540, "Capital_Growth_%": 7.1, "Zoning_Note": "High rental occupancy; strong family 2BR/3BR demand"},
    {"Region": "Kilimani / Kileleshwa", "Lat": -1.2896, "Lon": 36.7831, "Land_Per_Acre_KES": 380_000_000, "Build_Cost_Per_Sqm_KES": 68_000, "Sale_Price_Per_Sqm_KES": 105_000, "Monthly_Rent_Per_Sqm_KES": 780, "Capital_Growth_%": 5.8, "Zoning_Note": "High-rise executive apartments; higher finishing & lift specs"},
    {"Region": "Parklands / Westlands", "Lat": -1.2635, "Lon": 36.8124, "Land_Per_Acre_KES": 420_000_000, "Build_Cost_Per_Sqm_KES": 74_000, "Sale_Price_Per_Sqm_KES": 118_000, "Monthly_Rent_Per_Sqm_KES": 860, "Capital_Growth_%": 6.2, "Zoning_Note": "Premium residential hub; highest rental yield per sqm"},
    {"Region": "Syokimau / Mavoko (Mombasa Rd)", "Lat": -1.3612, "Lon": 36.9355, "Land_Per_Acre_KES": 68_000_000, "Build_Cost_Per_Sqm_KES": 46_000, "Sale_Price_Per_Sqm_KES": 58_000, "Monthly_Rent_Per_Sqm_KES": 360, "Capital_Growth_%": 8.2, "Zoning_Note": "Lower land entry cost; SGR/Expressway commuter corridor"},
    {"Region": "Ruaka / Kiambu Road", "Lat": -1.2118, "Lon": 36.7815, "Land_Per_Acre_KES": 135_000_000, "Build_Cost_Per_Sqm_KES": 50_000, "Sale_Price_Per_Sqm_KES": 66_000, "Monthly_Rent_Per_Sqm_KES": 450, "Capital_Growth_%": 7.8, "Zoning_Note": "UN/Two Rivers spillover demand; growing mid-income rental market"},
])

st.markdown("""
<div style="background: linear-gradient(90deg, #1b4f72 0%, #117a65 100%); padding: 18px 24px; border-radius: 10px; color: white; margin-bottom: 16px;">
    <h2 style="margin:0;">🏢 JengaApp — Nairobi Spatial Feasibility & Audit Portal</h2>
    <p style="margin:4px 0 0 0; opacity:0.92;">Click any Nairobi region on the map to simulate Land Cost, Construction CapEx, Tiered Unit Sales & Rental Yield</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🗺️ 1. Interactive Nairobi Map: Land, Build, Sale & Rent Simulator",
    "🛡️ 2. Live QuickBooks Ledger Fraud & Leakage Guard",
    "📊 3. JengaApp Phase I Empirical Baselines & Drift Monitor"
])

with tab1:
    map_col, ctrl_col = st.columns([1.45, 1.0])
    with map_col:
        st.subheader("Click a Nairobi Region Bubble to Load Market Benchmarks")
        fig_map = px.scatter_mapbox(
            NAIROBI_ZONES, lat="Lat", lon="Lon",
            size="Sale_Price_Per_Sqm_KES", color="Monthly_Rent_Per_Sqm_KES",
            hover_name="Region",
            hover_data={"Lat": False, "Lon": False, "Land_Per_Acre_KES": ":,.0f", "Build_Cost_Per_Sqm_KES": ":,.0f", "Sale_Price_Per_Sqm_KES": ":,.0f", "Monthly_Rent_Per_Sqm_KES": ":,.0f"},
            color_continuous_scale="Viridis", size_max=28, zoom=10.6,
            center={"lat": -1.285, "lon": 36.845}, mapbox_style="open-street-map", height=410
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points")

        clicked_region = None
        if map_event and map_event.get("selection") and map_event["selection"].get("points"):
            pt_idx = map_event["selection"]["points"][0]["point_index"]
            clicked_region = NAIROBI_ZONES.iloc[pt_idx]["Region"]

    with ctrl_col:
        st.subheader("Project & Area Specifications")
        default_idx = int(NAIROBI_ZONES[NAIROBI_ZONES["Region"] == clicked_region].index[0]) if clicked_region else 0
        selected_region = st.selectbox("Selected Nairobi Region (Click map or choose here)", NAIROBI_ZONES["Region"], index=default_idx)
        zone = NAIROBI_ZONES[NAIROBI_ZONES["Region"] == selected_region].iloc[0]
        st.info(f"**Market Profile:** {zone['Zoning_Note']}")

        plot_acres = st.selectbox("Land Plot Size (Acres)", [0.125, 0.25, 0.50, 1.00], index=1, format_func=lambda x: f"{x} Acres ({int(x*43560):,} sq ft)")
        unit_area_sqm = st.slider("Constructed Area per Apartment (sqm)", 65, 180, 110, step=5)
        num_floors = st.slider("Number of Floors", 6, 16, 13)
        units_per_floor = st.slider("Apartments per Floor", 4, 6, 5)
        floor_premium_pct = st.slider("Floor-Elevation Price Step-Up (%)", 0.0, 6.0, 3.5, step=0.5)
        occupancy_rate = st.slider("Expected Rental Occupancy (%)", 70, 100, 90, step=5) / 100.0

    total_units = num_floors * units_per_floor
    total_saleable_sqm = total_units * unit_area_sqm
    total_gross_built_sqm = total_saleable_sqm * 1.18

    est_land_cost = zone["Land_Per_Acre_KES"] * plot_acres
    est_construction_cost = total_gross_built_sqm * zone["Build_Cost_Per_Sqm_KES"]
    total_project_capex = est_land_cost + est_construction_cost
    all_in_cost_per_unit = total_project_capex / total_units

    floors = np.arange(1, num_floors + 1)
    tier_mult = np.where(floors <= num_floors / 3, 1.0, np.where(floors <= 2 * num_floors / 3, 1 + floor_premium_pct / 100.0, 1 + 2 * (floor_premium_pct / 100.0)))
    unit_sale_prices = (unit_area_sqm * zone["Sale_Price_Per_Sqm_KES"]) * tier_mult
    total_sales_revenue = (unit_sale_prices * units_per_floor).sum()
    net_sale_profit = total_sales_revenue - total_project_capex
    sale_margin_pct = (net_sale_profit / total_sales_revenue) * 100

    monthly_rent_per_unit = unit_area_sqm * zone["Monthly_Rent_Per_Sqm_KES"]
    net_annual_rental_income = (monthly_rent_per_unit * total_units * occupancy_rate * 12) * 0.85
    gross_rental_yield_pct = (net_annual_rental_income / total_project_capex) * 100
    payback_years = total_project_capex / max(1, net_annual_rental_income)

    st.markdown("---")
    st.subheader(f"Feasibility Summary: {selected_region} ({total_units} Units | {total_saleable_sqm:,.0f} Saleable sqm)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1. Est. Land Cost", f"KES {est_land_cost/1e6:,.1f}M", f"{plot_acres} Acres")
    m2.metric("2. Construction CapEx", f"KES {est_construction_cost/1e6:,.1f}M", f"KES {zone['Build_Cost_Per_Sqm_KES']:,}/sqm")
    m3.metric("3. Outright Sales Value", f"KES {total_sales_revenue/1e6:,.1f}M", f"Margin: {sale_margin_pct:.1f}% (+KES {net_sale_profit/1e6:.1f}M)")
    m4.metric("4. Monthly Rent / Unit", f"KES {monthly_rent_per_unit:,.0f}/mo", f"{unit_area_sqm} sqm @ KES {zone['Monthly_Rent_Per_Sqm_KES']}/sqm")
    m5.metric("5. Net Annual Rent & Yield", f"KES {net_annual_rental_income/1e6:,.2f}M/yr", f"{gross_rental_yield_pct:.2f}% Yield ({payback_years:.1f} yr payback)")

    ch1, ch2 = st.columns(2)
    with ch1:
        cap_df = pd.DataFrame({
            "Component": ["Land Acquisition", "Construction CapEx", "Total Project Cost", "Outright Sale Revenue"],
            "Amount_M": [est_land_cost/1e6, est_construction_cost/1e6, total_project_capex/1e6, total_sales_revenue/1e6],
            "Type": ["Cost", "Cost", "Total Cost", "Revenue"]
        })
        fig_cap = px.bar(cap_df, x="Component", y="Amount_M", color="Type", color_discrete_map={"Cost": "#e67e22", "Total Cost": "#c0392b", "Revenue": "#117a65"}, text_auto=".1f", title=f"Capital Stack vs. Sales Realization — {selected_region} (KES M)")
        fig_cap.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_cap, use_container_width=True)

    with ch2:
        fig_flr = go.Figure()
        fig_flr.add_trace(go.Bar(x=[f"Flr {f}" for f in floors], y=unit_sale_prices / 1e6, name="Unit Sale Price (KES M)", marker_color="#1b4f72"))
        fig_flr.add_trace(go.Scatter(x=[f"Flr {f}" for f in floors], y=np.repeat(all_in_cost_per_unit / 1e6, len(floors)), name="All-In Cost/Unit (Land+Build)", line=dict(color="#c0392b", dash="dash", width=2.5)))
        fig_flr.update_layout(title=f"Floor-Tiered Unit Sale Price vs. All-In Cost (Monthly Rent: KES {monthly_rent_per_unit:,.0f}/unit)", yaxis_title="KES Millions per Unit", height=360)
        st.plotly_chart(fig_flr, use_container_width=True)

with tab2:
    st.subheader("Automated Pre-Payment Voucher Audit & Leakage Scanner")
    uploaded_file = st.file_uploader("Upload Monthly QuickBooks Ledger (.xlsx)", type=["xlsx"])
    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        parsed_rows = []
        for sh in xls.sheet_names:
            if "wip" in sh.lower() or "work in progress" in sh.lower() or "depreciation" in sh.lower():
                continue
            raw = pd.read_excel(xls, sheet_name=sh, header=None)
            h_idx = next((i for i, r in raw.iterrows() if "date" in [str(v).strip().lower() for v in r.values if pd.notna(v)]), None)
            if h_idx is not None:
                df = raw.iloc[h_idx + 1:].copy()
                df.columns = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(raw.iloc[h_idx])]
                amt_col = next((c for c in df.columns if c.lower().startswith("amount")), None)
                if amt_col:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df["Amount"] = pd.to_numeric(df[amt_col], errors="coerce")
                    df = df[df["Date"].notna() & df["Amount"].notna()].copy()
                    df["ledger_key"] = "wip_incurred" if "incurred" in sh.lower() else sh.lower().replace(" ", "_")
                    df["entry_type"] = df["Type"].astype(str) if "Type" in df.columns else "General Journal"
                    df["memo_raw"] = df["Memo"].astype(str) if "Memo" in df.columns else ""
                    parsed_rows.append(df[["Date", "ledger_key", "entry_type", "memo_raw", "Amount"]])
        if parsed_rows:
            audit_df = pd.concat(parsed_rows, ignore_index=True)
            feats = bundle["feature_pipeline"].transform(audit_df)
            X_sc = bundle["scaler"].transform(feats[bundle["selected_features"]])
            audit_df["Audit_Risk_Score"] = bundle["audit_risk_gbm"].predict_proba(X_sc)[:, 1]
            audit_df["Is_Duplicate"] = audit_df.duplicated(subset=["ledger_key", "Date", "memo_raw", "Amount"], keep=False)
            audit_df["Action"] = np.where(audit_df["Is_Duplicate"], "BLOCK: Exact Duplicate", np.where(audit_df["Audit_Risk_Score"] >= 0.50, "HOLD: Require Invoice/CHQ Ref", "AUTO-CLEAR"))
            flagged = audit_df[audit_df["Action"] != "AUTO-CLEAR"].sort_values("Audit_Risk_Score", ascending=False)
            st.warning(f"Flagged {len(flagged):,} of {len(audit_df):,} vouchers for Finance Manager review.")
            st.dataframe(flagged.style.format({"Amount": "KES {:,.2f}", "Audit_Risk_Score": "{:.1%}"}), use_container_width=True)

with tab3:
    st.subheader("JengaApp Phase I Audited Reference & Model Drift Monitor")
    st.write(f"**Audited Phase I Unit Construction Cost (Excl. Land):** KES {bundle['baseline_unit_cost_kes']:,.2f} / unit (51 sold units = KES 323.87M Cost of Sales vs. KES 341.70M Sales Revenue).")
    st.dataframe(pd.DataFrame(bundle["psi_baseline"]), use_container_width=True)
