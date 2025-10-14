import pandas as pd
import itertools
import plotly.express as px
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
from copy import deepcopy

# Constants

client = EntsoePandasClient(api_key=st.secrets["api_keys"]["transparency_platform"])
print("Using TP client with key:", client.api_key[:8] + "...")
IC_PAIRS = {
    "IFA":  ("FR", "10Y1001C--00098F"),
    "IFA2": ("FR", "17Y0000009369493"),
    "Eleclink": ("FR", "11Y0-0000-0265-K"),
    "NemoLink": ("10YBE----------2", "10YGB----------A"),
    "BritNed":  ("10YNL----------L", "10YGB----------A"),
}

IC_PRETTY = {
    "IFA": "IFA 1",
    "IFA2": "IFA 2",
}

IC_LIMITS = {
    "IFA 1 node 1": {"export": 1000, "import": 1000},
    "IFA 1 node 2": {"export": 1000, "import": 1000},
    "IFA 2":        {"export": 1000, "import": 1000},
    "IFA 1":        {"export": 2000, "import": 2000},
    "Eleclink":     {"export": 1000, "import": 1000},
    "NemoLink":     {"export": 1000, "import": 1000},
    "BritNed":      {"export": 1000, "import": 1000},
}
DEFAULT_EXPORT_LIMIT = 1000.0
DEFAULT_IMPORT_LIMIT = 1000.0


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="HVDC Headroom", layout="wide")
st.sidebar.title("Date selection")
selected_date = st.sidebar.date_input("Select the delivery day", value="today", max_value=dt.date.today() + dt.timedelta(days=1))
date_string = selected_date.strftime("%Y%m%d")
model_choice = st.sidebar.selectbox(
    "Select model",
    options=["Day-Ahead", "Intraday"],
    index=0,
    help="Select based on the market you want to analyze (Day-Ahead or Intraday)."
)

overlay_da = False
if model_choice.upper() == "INTRADAY" and st.sidebar.checkbox("Show also day-ahead", value=False):
    overlay_da = True

# Build pretty labels from IC_PAIRS keys
available_nodes_pretty = [IC_PRETTY.get(k, k) for k in IC_PAIRS.keys()]

# Nice defaults if present, else select all
preferred_defaults = ["IFA 1", "IFA 2"]
default_nodes = [p for p in preferred_defaults if p in available_nodes_pretty]
if not default_nodes:
    default_nodes = ["IFA 1", "IFA 2"]  # fallback: select all


node_choice = st.sidebar.multiselect(
    "Choose interconnector(s)",
    options=available_nodes_pretty,
    default=default_nodes
)


# --- Map pretty -> key, and filter IC_PAIRS accordingly ---
PRETTY_TO_KEY = {v: k for k, v in IC_PRETTY.items()}

# If someone selects a raw key (no pretty), fall back gracefully
selected_keys = [PRETTY_TO_KEY.get(label, label) for label in node_choice]

filtered_ic_pairs = {k: IC_PAIRS[k] for k in selected_keys if k in IC_PAIRS}

# keep session state in sync
st.session_state.node_choice = node_choice

# Guardrail
if not node_choice:
    st.info("Select at least one interconnector to display results.")
    st.stop()


IC_LIMITS_DEFAULT = deepcopy(IC_LIMITS)

with st.sidebar.expander("Overwrite limits (MW)", expanded=False):
    step_mw = st.number_input("Slider step [MW]", min_value=10, max_value=500, value=50, step=10)


    ics_to_show = [ic for ic in IC_LIMITS_DEFAULT if ic in node_choice] or list(IC_LIMITS_DEFAULT.keys())
    new_limits = {}
    for ic in sorted(ics_to_show):
        defaults = IC_LIMITS_DEFAULT.get(ic, {"export": DEFAULT_EXPORT_LIMIT, "import": DEFAULT_IMPORT_LIMIT})

        # Per-IC slider range: from -import_default to +export_default
        min_val = int(-defaults.get("import", DEFAULT_IMPORT_LIMIT))
        max_val = int(+defaults.get("export", DEFAULT_EXPORT_LIMIT))
        default_range = (min_val, max_val)

        st.markdown(f"**{ic}**")
        left, right = st.slider(
            f"{ic} import/export range", key=f"{ic}-range",
            min_value=min_val, max_value=max_val,
            value=default_range, step=step_mw
        )

        import_limit = float(max(0, -left))   # left is negative
        export_limit = float(max(0, right))   # right is positive

        new_limits[ic] = {"export": export_limit, "import": import_limit}

    # Keep any ICs not shown so lookups never break
    for ic, vals in IC_LIMITS_DEFAULT.items():
        new_limits.setdefault(ic, vals)

# Replace the limits used downstream
IC_LIMITS = new_limits



if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

# Sidebar toggle
suppress_errors = st.sidebar.checkbox("Suppress API warnings", value=False)

def find_best_slot_cross(cross_df: pd.DataFrame, threshold: float = 100.0, offset_minutes: int = 0):
    """
    cross_df: wide DataFrame
        - Index: datetime (or a 'dt' column that will be set as index)
        - Columns: pair names like 'IFA↑ IFA 2↓', values = headroom MW
    threshold: MW threshold the series must be >= to qualify
    offset_minutes: subtract from the start time (your 30-min convention)

    Returns: (best_time, best_pair, longest_duration_hours)
    """
    if cross_df is None or cross_df.empty:
        return None, None, 0.0

    df = cross_df.copy()
    if "dt" in df.columns:
        df = df.set_index("dt")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None, None, 0.0
    df = df[numeric_cols]

    mask = df >= threshold

    # infer slot length (fallback 30 min)
    idx_diff = df.index.to_series().diff().dropna()
    slot = idx_diff.mode().iloc[0] if not idx_diff.empty else pd.Timedelta(minutes=30)

    longest_hours = 0.0
    best_start = None
    best_pair = None

    for col in mask.columns:
        col_mask = mask[col]
        if col_mask.isna().all():
            continue

        seg_id = (col_mask != col_mask.shift()).cumsum()

        for sid in seg_id.unique():
            idx = df.index[seg_id == sid]
            if len(idx) == 0:
                continue
            if not col_mask.loc[idx[0]]:
                continue

            start, end = idx[0], idx[-1]
            # include the width of the last slot
            duration_hours = ((end - start) + slot).total_seconds() / 3600.0

            if duration_hours > longest_hours:
                longest_hours = duration_hours
                best_start = start
                best_pair = col

    if best_start is not None:
        best_start = best_start - timedelta(minutes=offset_minutes)

    return best_start, best_pair, longest_hours

@st.cache_data(show_spinner=False, ttl=30) 
def get_tp_net_schedules(
    date_string: dt.date | str,
    model_choice: str,  # "Day-Ahead" -> DA, "Intraday" -> total/intraday
    ic_pairs: dict[str, tuple[str, str]],
    tz: str = "Europe/Brussels",
    pretty_names: dict[str, str] | None = None,
    auto_capacity: bool = False,
):
    """
    Returns df (long): dt, node, pretty, p_active, model
    If auto_capacity=True, also returns capa_dict with, per IC:
        {"export": Series(base->cp DA NTC), "import": Series(cp->base DA NTC)}
    """
    if isinstance(date_string, dt.date):
        start = pd.Timestamp(date_string, tz=tz)
    else:
        start = pd.Timestamp(str(date_string), tz=tz)
    end = start + pd.Timedelta(days=1)

    is_dayahead = (model_choice.upper() == "DAY-AHEAD")
    model_tag = "Day-Ahead" if is_dayahead else "Intraday"

    def safe_query(qname, func, *args, **kwargs):
        try:
            print(f"🔍 Querying: {qname}")
            return func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ Error during query: {qname}")
            if not suppress_errors:
                st.warning(f"Could not fetch data for {qname}. See console for details.")
            print("   Exception:", e)
            return pd.Series(dtype=float)

    records = []
    capa_dict: dict[str, dict[str, pd.Series]] = {}

    for node, (base_eic, cp_eic) in ic_pairs.items():
        # Scheduled exchanges (net = base->cp - cp->base)
        a_to_b = safe_query(
            f"{base_eic} → {cp_eic} {'DA' if is_dayahead else 'total'}",
            client.query_scheduled_exchanges,
            base_eic, cp_eic, start=start, end=end, dayahead=is_dayahead
        )
        b_to_a = safe_query(
            f"{cp_eic} → {base_eic} {'DA' if is_dayahead else 'total'}",
            client.query_scheduled_exchanges,
            cp_eic, base_eic, start=start, end=end, dayahead=is_dayahead
        )
        net = a_to_b.subtract(b_to_a, fill_value=0)

        if not isinstance(net.index, pd.DatetimeIndex):
            net.index = pd.to_datetime(net.index)
        df_node = net.rename("p_active").to_frame()
        df_node["dt"] = df_node.index.tz_convert(None) if df_node.index.tz is not None else df_node.index
        df_node["node"] = node
        df_node["pretty"] = pretty_names.get(node, node) if pretty_names else node
        df_node["model"] = model_tag
        records.append(df_node[["dt", "node", "pretty", "p_active", "model"]])

        # Optional DA NTC in BOTH directions
        if auto_capacity:
            cap_export = safe_query(
                f"{base_eic} → {cp_eic} capacity (DA)",
                client.query_net_transfer_capacity_dayahead,
                base_eic, cp_eic, start=start, end=end
            )
            cap_import = safe_query(
                f"{cp_eic} → {base_eic} capacity (DA)",
                client.query_net_transfer_capacity_dayahead,
                cp_eic, base_eic, start=start, end=end
            )
            capa_dict[node] = {"export": cap_export, "import": cap_import}

    df_all = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["dt","node","pretty","p_active","model"])
    if not df_all.empty:
        df_all["dt"] = pd.to_datetime(df_all["dt"]).dt.tz_localize(None)
        df_all["pretty"] = df_all["pretty"].astype("category")
        df_all["node"] = df_all["node"].astype("category")

    return (df_all, capa_dict) if auto_capacity else df_all


# --- Fetch data & capacities (both directions) ---
df, capa_dict = get_tp_net_schedules(
    date_string=date_string,
    model_choice=model_choice,
    ic_pairs=filtered_ic_pairs,
    tz="Europe/Brussels",
    pretty_names=IC_PRETTY,
    auto_capacity=True
)

# Optional Day-Ahead overlay when viewing Intraday
df_da = pd.DataFrame()
if overlay_da:
    df_da = get_tp_net_schedules(
        date_string=date_string,
        model_choice="DAY-AHEAD",
        ic_pairs=filtered_ic_pairs,
        tz="Europe/Brussels",
        pretty_names=IC_PRETTY,
        auto_capacity=False
    )

# --- Build a capacity dataframe with BOTH directions (no model column; applies to any view) ---
capa_rows = []
for ic, d in (capa_dict or {}).items():
    # export = base->cp, import = cp->base
    for kind in ("export", "import"):
        ser = d.get(kind, pd.Series(dtype=float))
        if ser is None or ser.empty:
            continue
        s = ser.rename(f"cap_{kind}").to_frame()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s["dt"] = s.index.tz_convert(None) if s.index.tz is not None else s.index
        s["pretty"] = IC_PRETTY.get(ic, ic)
        capa_rows.append(s[["dt", "pretty", f"cap_{kind}"]])

# one row per dt/pretty with possible cap_export and/or cap_import
df_capacity = (pd.concat(capa_rows, ignore_index=True)
               .groupby(["dt","pretty"], as_index=False).max()
               if capa_rows else pd.DataFrame(columns=["dt","pretty","cap_export","cap_import"]))

# --- Compose the working df (current view + optional DA overlay) ---
work = df.copy() if df_da.empty else pd.concat([df, df_da], ignore_index=True)

# Build an empty capacity frame with required columns when no rows
if df_capacity is None or df_capacity.empty:
    df_capacity = pd.DataFrame(columns=["dt", "pretty", "cap_export", "cap_import"])

# Always merge so cap_* columns exist (they'll be NaN if no data)
work = work.merge(df_capacity, on=["dt", "pretty"], how="left")


# Start from slider/defaults (same as you had)
work["limit_export"] = work["pretty"].map(lambda k: IC_LIMITS.get(k, {}).get("export", DEFAULT_EXPORT_LIMIT))
work["limit_import"] = work["pretty"].map(lambda k: IC_LIMITS.get(k, {}).get("import", DEFAULT_IMPORT_LIMIT))

# Override with TP capacities when available (directional!)
# First, ensure numeric dtypes for all relevant columns
for col in ["p_active", "cap_export", "cap_import", "limit_export", "limit_import"]:
    if col not in work.columns:
        work[col] = pd.NA
    work[col] = pd.to_numeric(work[col], errors="coerce")

has_cap_exp = work["cap_export"].notna()
has_cap_imp = work["cap_import"].notna()
work.loc[has_cap_exp, "limit_export"] = work.loc[has_cap_exp, "cap_export"]
work.loc[has_cap_imp, "limit_import"] = work.loc[has_cap_imp, "cap_import"]

# Now safe to do arithmetic
work["hr_export_mw"] = (work["limit_export"] - work["p_active"]).clip(lower=0)
work["hr_import_mw"] = (work["p_active"] + work["limit_import"]).clip(lower=0)

# --- The rest of your plotting / tables can remain unchanged ---
# e.g.:
st.title(f"HVDC Headroom — {selected_date} ({model_choice})")
st.caption("Schedules from ENTSO-E TP; limits overridden by directional DA NTC when available.")


# -----------------------------
# PLOTS & TABLES (TP-based)
# -----------------------------

# Filter the active model by interconnector selection
fdf = df[df["pretty"].isin(node_choice)].copy()

# Optional overlay: concat Intraday+Day-Ahead (when overlay enabled)
fdf_all = fdf.copy() if df_da.empty else pd.concat(
    [fdf, df_da[df_da["pretty"].isin(node_choice)].copy()],
    ignore_index=True
)

# Series label like before: "<interconnector> (MODEL)"
fdf_all["series"] = fdf_all["pretty"].astype(str) + " (" + fdf_all["model"].astype(str) + ")"

# Pivot for schedule table (no 'file' available from TP)
wide_df = (
    fdf.pivot(index=["dt"], columns="pretty", values="p_active")
       .sort_index()
       .reset_index()
       .rename(columns={"dt": "Datetime"})
)

# Limits already overridden by TP capacities in your 'work' df.
# Keep consistent selection (and overlay) for headroom plots/tables:
work_view = work[work["pretty"].isin(node_choice)].copy()
work_view["series"] = work_view["pretty"].astype(str) + " (" + work_view["model"].astype(str) + ")"

models_present = sorted(work_view["model"].unique().tolist())
if models_present == []:
    st.error("No data available for the selected date/interconnector/model.")
title_models = " vs ".join(models_present) if len(models_present) > 1 else models_present[0]

# Wide headroom tables
hr_export_wide = work_view.pivot_table(index=["dt","model"], columns="pretty", values="hr_export_mw").sort_index()
hr_import_wide = work_view.pivot_table(index=["dt","model"], columns="pretty", values="hr_import_mw").sort_index()

max_export_limit = float(work_view["limit_export"].max()) if not work_view.empty else 0.0
max_import_limit = float(work_view["limit_import"].max()) if not work_view.empty else 0.0

# -----------------------------
# Cross-headroom (pairs)
# -----------------------------
if len(node_choice) < 2:
    st.warning(
        f"Cross-redirection needs at least two interconnectors. "
        f"You selected only **{node_choice[0]}** — skipping cross-headroom search, chart and table."
    )
    cross_long_all = pd.DataFrame()
else:
    pair_frames = []
    for the_model, g in work_view.groupby("model"):
        tmp = g.copy()
        sub_pairs = []
        for a, b in itertools.combinations(tmp["pretty"].unique(), 2):
            fa = tmp[tmp["pretty"] == a].set_index("dt")["hr_export_mw"].rename(f"hr_export_mw_{a}")
            fa_imp = tmp[tmp["pretty"] == a].set_index("dt")["hr_import_mw"].rename(f"hr_import_mw_{a}")
            fb = tmp[tmp["pretty"] == b].set_index("dt")["hr_export_mw"].rename(f"hr_export_mw_{b}")
            fb_imp = tmp[tmp["pretty"] == b].set_index("dt")["hr_import_mw"].rename(f"hr_import_mw_{b}")
            joined = pd.concat([fa, fa_imp, fb, fb_imp], axis=1, join="inner")
            joined[f"{a}↑ {b}↓"] = joined[f"hr_export_mw_{a}"].combine(joined[f"hr_import_mw_{b}"], min)
            joined[f"{a}↓ {b}↑"] = joined[f"hr_import_mw_{a}"].combine(joined[f"hr_export_mw_{b}"], min)
            sub_pairs.append(joined[[f"{a}↑ {b}↓", f"{a}↓ {b}↑"]])
        if sub_pairs:
            cross_df_model = pd.concat(sub_pairs, axis=1)
            cross_long_model = (
                cross_df_model.reset_index()
                              .melt(id_vars="dt", var_name="pair", value_name="headroom_mw")
            )
            cross_long_model["model"] = the_model
            pair_frames.append(cross_long_model)

    cross_long_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()

    # Business-hour scan first (08–17 local). Model strings are "Intraday" / "Day-Ahead".
    ymax_cross = float(min(max_export_limit, max_import_limit)) if max_export_limit and max_import_limit else None
    def _best_slot_from_long(df_long):
        if df_long.empty:
            return None, None, 0.0
        wide = df_long.pivot(index="dt", columns="pair", values="headroom_mw").sort_index()
        return find_best_slot_cross(wide, threshold=100)

    # --- Choose dataset based on selected model ---
    # model_choice is your sidebar selectbox value: "Day-Ahead" or "Intraday"
    model_tag = "Day-Ahead" if model_choice.upper() == "DAY-AHEAD" else "Intraday"

    # Business hours and preferred morning window
    BUSINESS_START, BUSINESS_END = 9, 16   # inclusive start, inclusive end-1 (i.e. 08:00–16:59)
    PREFERRED_END = 12                     # 08:00–11:59 is the preferred window

    # Filter to the selected model only
    df_model = cross_long_all[cross_long_all["model"] == model_tag].copy()

    # 1) Try preferred morning window (08:00–12:00)
    morning_mask = df_model["dt"].dt.hour.between(BUSINESS_START, PREFERRED_END - 1)
    best_time, best_pair, longest_duration = _best_slot_from_long(df_model[morning_mask])

    # 2) If none, try full business hours (08:00–17:00)
    if best_time is None:
        biz_mask = df_model["dt"].dt.hour.between(BUSINESS_START, BUSINESS_END)
        best_time, best_pair, longest_duration = _best_slot_from_long(df_model[biz_mask])

    # 3) If still none, try the whole day (for the selected model)
    if best_time is None:
        best_time, best_pair, longest_duration = _best_slot_from_long(df_model)

    # --- Banner logic per your spec ---
    if best_time is not None:
        h = best_time.hour
        in_morning_pref = (BUSINESS_START <= h < PREFERRED_END)
        in_business_hours = (BUSINESS_START <= h < BUSINESS_END)

        if in_morning_pref:
            st.success(
                f"Best slot for a trial starts at **{best_time.strftime('%Y-%m-%d %H:%M')}** "
                f"for **{longest_duration:.0f} hours** with ≥100 MW on pair **{best_pair}** "
                f"(within **{BUSINESS_START:2d}:00–{BUSINESS_END}:00**)."
            )
        elif not in_business_hours:
            st.warning(
                f"Best slot for a trial starts at **{best_time.strftime('%Y-%m-%d %H:%M')}** "
                f"for **{longest_duration:.0f} hours** with ≥100 MW on pair **{best_pair}**, "
                f"but it’s **outside business hours ({BUSINESS_START}:00–{BUSINESS_END}:00)**."
            )
        else:
            # Within business hours but after 12:00
            st.success(
                f"Best slot for a trial starts at **{best_time.strftime('%Y-%m-%d %H:%M')}** "
                f"for **{longest_duration:.0f} hours** with ≥100 MW on pair **{best_pair}** "
                f"(within business hours, after 12:00)."
            )
    else:
        st.info(f"No slot found with ≥100 MW headroom on any pair for **{model_tag}**.")


    if not cross_long_all.empty:
        fig = px.bar(
            cross_long_all,
            x="dt", y="headroom_mw",
            color="model",
            facet_row="pair",
            barmode="group",
            title=f"Redirection availability over time ({title_models})",
            labels={"dt":"Datetime","headroom_mw":"Headroom [MW]","model":"Model"},
        )
        if ymax_cross is not None:
            fig.update_yaxes(range=[0, ymax_cross], matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Redirection availability data table", expanded=False):
            st.dataframe(
                cross_long_all.pivot_table(index="dt", columns=["model","pair"], values="headroom_mw"),
                use_container_width=True, hide_index=True
            )

# -----------------------------
# Individual headroom analysis
# -----------------------------
with st.expander("Individual headroom analysis", expanded=False):
    st.subheader("Export headroom to +limit (per interconnector)")
    st.dataframe(
        hr_export_wide.reset_index().rename(columns={"dt":"Datetime"}),
        use_container_width=True, hide_index=True
    )
    st.subheader("Import headroom to -limit (per interconnector)")
    st.dataframe(
        hr_import_wide.reset_index().rename(columns={"dt":"Datetime"}),
        use_container_width=True, hide_index=True
    )

    fig_exp = px.line(
        work_view, x="dt", y="hr_export_mw", color="series",
        title="Headroom to +limit (Exports)",
        line_shape="hv",
        labels={"dt":"Datetime","hr_export_mw":"Headroom [MW]","series":"Series"},
    )
    if max_export_limit:
        fig_exp.update_yaxes(range=[0, max_export_limit])
    st.plotly_chart(fig_exp, use_container_width=True)

    fig_imp = px.line(
        work_view, x="dt", y="hr_import_mw", color="series",
        title="Headroom to -limit (Imports)",
        line_shape="hv",
        labels={"dt":"Datetime","hr_import_mw":"Headroom [MW]","series":"Series"},
    )
    if max_import_limit:
        fig_imp.update_yaxes(range=[0, max_import_limit])
    st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------
# Net schedules (from TP)
# -----------------------------
fig = px.line(
    fdf_all,
    x="dt",
    y="p_active",
    color="series",
    title=f"Net Schedule by Interconnector ({title_models})",
    line_shape="hv"
)
fig.update_layout(
    xaxis_title="Datetime",
    yaxis_title="MW",
    legend_title="Series",
)

# --- Overlay directional capacities on the net schedules plot ---

# Capacities are the same across models; collapse to one row per dt/pretty
cap_lines = (
    work_view
    .groupby(["dt", "pretty"], as_index=False)[["limit_export", "limit_import"]]
    .max()
)

# Add a dashed +capacity and -capacity line for each selected interconnector
for ic in sorted(cap_lines["pretty"].unique()):
    sub = cap_lines[cap_lines["pretty"] == ic].sort_values("dt").copy()

    # Clean NaNs just in case
    y_plus = sub["limit_export"].astype("float")
    y_minus = (-sub["limit_import"].astype("float"))

    # +capacity (export limit)
    fig.add_scatter(
        x=sub["dt"],
        y=y_plus,
        mode="lines",
        name=f"{ic} +capacity",
        line=dict(dash="dash"),
        line_shape="hv",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>+capacity: %{y:.0f} MW<extra></extra>",
        legendgroup=f"{ic}_cap",
    )

    # -capacity (import limit)
    fig.add_scatter(
        x=sub["dt"],
        y=y_minus,
        mode="lines",
        name=f"{ic} -capacity",
        line=dict(dash="dash"),
        line_shape="hv",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>-capacity: %{y:.0f} MW<extra></extra>",
        legendgroup=f"{ic}_cap",
    )

# Optional: clarify axes/legend
fig.update_layout(
    yaxis_title="MW (net schedule & capacities)",
    legend_title="Series / Capacities",
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show net schedule data table", expanded=False):
    table_df = (
        fdf_all.pivot(index=["dt","model"], columns="pretty", values="p_active")
              .sort_index()
              .reset_index()
              .rename(columns={"dt":"Datetime"})
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

# CSV export of the filtered + overlay data
csv = fdf_all.sort_values(["dt","pretty","model"]).to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", csv, "hvdc_filtered.csv", "text/csv")
