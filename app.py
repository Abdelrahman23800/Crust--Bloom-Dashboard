# crust_bloom_dashboard.py
# Final Streamlit dashboard for Crust & Bloom (complete, robust version)
# Expected columns (case-insensitive): Date, Time, Product Type, Units Produced, Units Sold, Revenue, Waste,
# Customer Type, Ad Campaign Source, Ad Spend, Revenue Attributed to Each Campaign
#
# How to run:
# 1. Save as crust_bloom_dashboard.py
# 2. pip install streamlit pandas plotly
# 3. streamlit run crust_bloom_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Crust & Bloom — Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

# ----------------- Utilities -----------------
def normalize_cols(df):
    """
    Return dataframe and a lookup dict mapping lowercase stripped colname -> original column name.
    """
    df = df.copy()
    lookup = {c.strip().lower(): c for c in df.columns}
    # keep original exact names but stripped
    df.columns = [c.strip() for c in df.columns]
    return df, lookup

def find_col(lookup, candidates):
    """
    Given lookup dict and list of lowercase candidate strings,
    return the original column name if a candidate is found (first match).
    """
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None

import os

@st.cache_data
def load_file(uploaded_file=None, local_path=None):
    """
    Load dataframe from uploaded file or local path.
    Returns (df, lookup) or (None, None) on failure.
    This version avoids printing st.error for missing local files;
    it only returns None so caller can handle UI messaging.
    """
    try:
        if uploaded_file is not None:
            # uploaded_file can be a streamlit UploadedFile or a path string
            if isinstance(uploaded_file, str):
                if uploaded_file.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
            else:
                # UploadedFile object
                name = getattr(uploaded_file, "name", "")
                if name.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
        else:
            # If local_path provided, check existence first
            if local_path and os.path.exists(local_path):
                df = pd.read_excel(local_path)
            else:
                # local file not provided or doesn't exist — return None to caller
                return None, None
    except Exception as e:
        # Don't call st.error here (to avoid UI error box inside cached function).
        # Return None and let caller show a friendly message once (non-cached).
        return None, None

    df, lookup = normalize_cols(df)
    # try coerce date if present
    date_col = find_col(lookup, ['date'])
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except:
            pass
    return df, lookup


def safe_sum(series):
    try:
        return float(pd.to_numeric(series, errors='coerce').sum())
    except:
        return 0.0

def df_to_csv_bytes(df):
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception:
        return b""

def text_report(kpis, prod_summary, campaign_summary, meta):
    lines = []
    lines.append("Crust & Bloom — Executive Summary")
    lines.append("="*72)
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("KEY KPIs")
    lines.append(f"- Total Revenue: ${kpis.get('total_revenue',0):,.2f}")
    lines.append(f"- Total Units Sold: {int(kpis.get('total_units_sold',0)):,}")
    lines.append(f"- Waste Rate: {kpis.get('waste_rate',0)*100:.2f}%")
    lines.append(f"- Avg Price / Unit: ${kpis.get('avg_price',0):.2f}")
    lines.append("")
    if campaign_summary is not None and not campaign_summary.empty:
        lines.append("TOP CAMPAIGNS (by Revenue)")
        for _, r in campaign_summary.head(5).iterrows():
            lines.append(f" - {r.get('campaign','N/A')}: Revenue ${r.get('revenue',0):,.2f} | AdSpend ${r.get('ad_spend',0):,.2f} | ROI {r.get('roi_str','N/A')}")
        lines.append("")
    if prod_summary is not None and not prod_summary.empty:
        lines.append("TOP PRODUCTS (by Revenue)")
        for _, r in prod_summary.head(5).iterrows():
            lines.append(f" - {r.get('product','N/A')}: Units Sold {int(r.get('units_sold',0))}, Revenue ${r.get('revenue',0):,.2f} | Waste Rate {r.get('waste_rate_pct','N/A')}")
    lines.append("")
    lines.append(f"Data source: {meta}")
    lines.append("="*72)
    return "\n".join(lines)

# ----------------- UI: header & load -----------------
st.title("Crust & Bloom — Sales & Campaigns Dashboard")
st.markdown("Analyze product performance, waste, and ad campaign ROI. Use sidebar filters to explore the data.")

st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload Excel / CSV (optional)", type=['xlsx','xls','csv'])
# default local path (the file you uploaded earlier)
local_path_default = "/mnt/data/01K8G7W0DMB43Q8VQ957YBSFKA.xlsx"
use_local_if_no_upload = st.sidebar.checkbox("If no upload, use local sample file", value=True)

# Decide source without causing errors if file missing
df_lookup = (None, None)
if uploaded is not None:
    df_lookup = load_file(uploaded_file=uploaded, local_path=None)
elif use_local_if_no_upload:
    # check file existence before trying to load
    if os.path.exists(local_path_default):
        df_lookup = load_file(uploaded_file=None, local_path=local_path_default)
    else:
        # local file not found — keep df_lookup as (None, None) and show info to user
        df_lookup = (None, None)

df, lookup = df_lookup if df_lookup is not None else (None, None)

if df is None:
    st.stop()

# ----------------- Auto-detect columns -----------------
product_col = find_col(lookup, ['product type','product','product_type','productname','item'])
units_sold_col = find_col(lookup, ['units sold','units_sold','sold','quantity','qty'])
units_prod_col = find_col(lookup, ['units produced','units_produced','produced'])
revenue_col = find_col(lookup, ['revenue','sales','total revenue','total'])
waste_col = find_col(lookup, ['waste','wasted','loss'])
customer_col = find_col(lookup, ['customer type','customer_type','customer'])
campaign_col = find_col(lookup, ['ad campaign source','ad_campaign_source','campaign','channel','adcampaign','ad campaign'])
ad_spend_col = find_col(lookup, ['ad spend','ad_spend','ad_spending','adcost'])
rev_attr_col = find_col(lookup, ['revenue attributed to each campaign','revenue attributed','revenue_attributed','revenue_attributed_to_campaign','rev_attr'])

# Show detected columns in sidebar

st.sidebar.markdown("**Detected columns (case-insensitive match):**")
st.sidebar.write(f"- Product Type: `{product_col}`")
st.sidebar.write(f"- Units Sold: `{units_sold_col}`")
st.sidebar.write(f"- Units Produced: `{units_prod_col}`")
st.sidebar.write(f"- Revenue: `{revenue_col}`")
st.sidebar.write(f"- Waste: `{waste_col}`")
st.sidebar.write(f"- Customer Type: `{customer_col}`")
st.sidebar.write(f"- Ad Campaign: `{campaign_col}`")
st.sidebar.write(f"- Ad Spend: `{ad_spend_col}`")
st.sidebar.write(f"- Revenue Attributed: `{rev_attr_col}`")


# Warn if core columns missing
if product_col is None or units_sold_col is None or revenue_col is None:
    st.warning("Core columns missing (Product Type, Units Sold, Revenue). Some visuals may be unavailable or incomplete.")

# ----------------- Filters (adaptive by column) -----------------
st.sidebar.header("Filters")

# Date range filter
date_min = df[find_col(lookup, ['date'])] if find_col(lookup, ['date']) else None
if date_min is not None:
    date_col_name = find_col(lookup, ['date'])
    min_d = df[date_col_name].min().date()
    max_d = df[date_col_name].max().date()
    date_range = st.sidebar.date_input("Date Range", value=[min_d, max_d], min_value=min_d, max_value=max_d)
    if len(date_range) == 2:
        df = df[(df[date_col_name] >= pd.to_datetime(date_range[0])) & (df[date_col_name] <= pd.to_datetime(date_range[1]))]

# Product Type filter (multi-select)


# Customer Type filter (multi-select)
if customer_col:
    cust_options = df[customer_col].dropna().unique().tolist()
    sel_cust = st.sidebar.multiselect("Customer Type", options=cust_options, default=cust_options)
    if sel_cust:
        df = df[df[customer_col].isin(sel_cust)]

# Campaign filter (multi-select)
if campaign_col:
    camp_options = df[campaign_col].dropna().unique().tolist()
    sel_camp = st.sidebar.multiselect("Ad Campaign Source", options=camp_options, default=camp_options)
    if sel_camp:
        df = df[df[campaign_col].isin(sel_camp)]

# ----------------- Coerce numeric columns safely -----------------
if units_sold_col:
    df[units_sold_col] = pd.to_numeric(df[units_sold_col], errors='coerce').fillna(0)
if units_prod_col:
    df[units_prod_col] = pd.to_numeric(df[units_prod_col], errors='coerce').fillna(0)
if revenue_col:
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
if waste_col:
    df[waste_col] = pd.to_numeric(df[waste_col], errors='coerce').fillna(0)
if ad_spend_col:
    df[ad_spend_col] = pd.to_numeric(df[ad_spend_col], errors='coerce').fillna(0)
if rev_attr_col:
    df[rev_attr_col] = pd.to_numeric(df[rev_attr_col], errors='coerce').fillna(0)

# ----------------- KPIs -----------------
total_revenue = safe_sum(df[revenue_col]) if revenue_col else 0.0
total_units = int(safe_sum(df[units_sold_col])) if units_sold_col else 0
total_produced = int(safe_sum(df[units_prod_col])) if units_prod_col else 0
total_waste = safe_sum(df[waste_col]) if waste_col else 0.0
waste_rate = (total_waste / total_produced) if total_produced > 0 else (total_waste / total_units if total_units > 0 else 0.0)
avg_price = (total_revenue / total_units) if total_units > 0 else 0.0

# Campaign summary & ROI
if campaign_col:
    # choose revenue attribution column if present, else use overall revenue grouped by campaign
    if rev_attr_col:
        camp_df = df.groupby(campaign_col).agg(ad_spend=(ad_spend_col, 'sum') if ad_spend_col else (rev_attr_col, 'count'),
                                               revenue=(rev_attr_col, 'sum')).reset_index()
    else:
        camp_df = df.groupby(campaign_col).agg(ad_spend=(ad_spend_col, 'sum') if ad_spend_col else (revenue_col, 'count'),
                                               revenue=(revenue_col, 'sum')).reset_index()
    def compute_roi(row):
        ad = float(row.get('ad_spend', 0) or 0)
        rev = float(row.get('revenue', 0) or 0)
        if ad > 0:
            return (rev - ad) / ad
        else:
            return np.nan
    if not camp_df.empty:
        camp_df['roi'] = camp_df.apply(compute_roi, axis=1)
        camp_df['roi_str'] = camp_df['roi'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        camp_df = camp_df.sort_values('revenue', ascending=False)
else:
    camp_df = pd.DataFrame(columns=['campaign','ad_spend','revenue','roi','roi_str'])

# Render KPI cards
k1, k2, k3, k4 = st.columns([1.5,1,1,1])
k1.metric("Total Revenue", f"${total_revenue:,.2f}")
k2.metric("Units Sold", f"{total_units:,}")
k3.metric("Waste Rate", f"{waste_rate*100:.2f}%")
k4.metric("Avg Price / Unit", f"${avg_price:.2f}")

st.markdown("---")

# ----------------- Product Performance -----------------
st.subheader("Product Performance")
if product_col and units_sold_col and revenue_col:
    prod = df.groupby(product_col).agg(units_sold=(units_sold_col,'sum'),
                                       revenue=(revenue_col,'sum'),
                                       units_produced=(units_prod_col,'sum') if units_prod_col else (units_sold_col,'sum'),
                                       waste=(waste_col,'sum') if waste_col else (units_sold_col,'sum')).reset_index()
    if 'units_produced' in prod.columns and prod['units_produced'].sum() > 0:
        prod['waste_rate_pct'] = (prod['waste'] / prod['units_produced']).round(4).apply(lambda x: f"{x*100:.2f}%")
    prod = prod.sort_values('revenue', ascending=False).reset_index(drop=True)
    fig_prod = px.bar(prod.head(15), x=product_col, y='revenue', hover_data=['units_sold','waste','units_produced'], title='Top products by revenue')
    st.plotly_chart(fig_prod, use_container_width=True)
    # show dataframe with clear names
    display_prod = prod.rename(columns={product_col:'product','units_sold':'Units Sold','revenue':'Revenue','units_produced':'Units Produced','waste':'Waste'})
    st.dataframe(display_prod.head(50).round(2))
else:
    st.info("Product performance requires Product Type, Units Sold and Revenue columns.")

st.markdown("---")

# ----------------- Time Trend -----------------
st.subheader("Revenue Trend")
date_col_name = find_col(lookup, ['date'])
if date_col_name and revenue_col:
    df_time = df.set_index(date_col_name).resample('W')[revenue_col].sum().reset_index()
    fig_time = px.line(df_time, x=date_col_name, y=revenue_col, title='Weekly Revenue Trend', markers=True)
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("Date or Revenue column missing for time trends.")

st.markdown("---")

# ----------------- Campaign Performance & ROI -----------------
st.subheader("Campaign Performance & ROI")
if campaign_col and not camp_df.empty:
    fig_camp = px.bar(camp_df, x=campaign_col, y='revenue', hover_data=['ad_spend','roi'], title='Campaign Revenue (hover: ad spend & ROI)')
    st.plotly_chart(fig_camp, use_container_width=True)
    display_camp = camp_df.rename(columns={campaign_col:'campaign','ad_spend':'Ad Spend','revenue':'Revenue','roi':'ROI'})
    st.dataframe(display_camp.round(3))
else:
    st.info("No campaign data found. Ensure 'Ad Campaign Source' or similar column exists.")

st.markdown("---")

# ----------------- Waste Analysis -----------------
st.subheader("Waste Analysis")
if units_prod_col and waste_col and product_col:
    scatter = df.groupby(product_col).agg(units_produced=(units_prod_col,'sum'), waste=(waste_col,'sum')).reset_index()
    scatter['waste_pct'] = (scatter['waste'] / scatter['units_produced']).fillna(0)
    fig_waste = px.scatter(scatter, x='units_produced', y='waste', size='waste_pct', hover_name=product_col, title='Waste vs Production (size = waste%)')
    st.plotly_chart(fig_waste, use_container_width=True)
    st.dataframe(scatter.head(50).round(3))
else:
    st.info("Waste analysis needs Units Produced and Waste columns.")

st.markdown("---")

# ----------------- Downloads (robust) -----------------
st.subheader("Export / Download")
meta = f"Rows: {len(df)} | Period: {df[date_col_name].min().date() if date_col_name in df.columns else 'N/A'} to {df[date_col_name].max().date() if date_col_name in df.columns else 'N/A'}"
kpis = {
    'total_revenue': total_revenue,
    'total_units_sold': total_units,
    'waste_rate': waste_rate,
    'avg_price': avg_price
}

# Ensure product & campaign DataFrames exist
if 'prod' not in locals():
    prod = pd.DataFrame(columns=[product_col or 'product', 'units_sold', 'revenue'])
if 'camp_df' not in locals() or camp_df is None:
    camp_df = pd.DataFrame(columns=[campaign_col or 'campaign', 'ad_spend', 'revenue', 'roi', 'roi_str'])

csv_prod = df_to_csv_bytes(prod) if prod is not None else b""
csv_camp = df_to_csv_bytes(camp_df) if camp_df is not None else b""
report_txt_str = text_report(kpis, prod if prod is not None else pd.DataFrame(), camp_df if camp_df is not None else pd.DataFrame(), meta)
report_txt = report_txt_str.encode('utf-8')

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    if csv_prod and len(csv_prod) > 0:
        st.download_button("Download Product Summary (CSV)", data=csv_prod, file_name="product_summary.csv", mime="text/csv")
    else:
        st.download_button("Download Product Summary (CSV)", data=csv_prod, file_name="product_summary.csv", mime="text/csv", disabled=True, help="No product summary available to download.")
with col_dl2:
    if csv_camp and len(csv_camp) > 0:
        st.download_button("Download Campaign Summary (CSV)", data=csv_camp, file_name="campaign_summary.csv", mime="text/csv")
    else:
        st.download_button("Download Campaign Summary (CSV)", data=csv_camp, file_name="campaign_summary.csv", mime="text/csv", disabled=True, help="No campaign summary available to download.")
with col_dl3:
    st.download_button("Download Executive Summary (TXT)", data=report_txt, file_name="crust_bloom_executive_summary.txt", mime="text/plain")

