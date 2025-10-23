# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import StringIO

st.set_page_config(layout="wide", page_title="Аналiтика трафiку вебсайту")

st.title("Аналітика трафіку вебсайту — CSV / Google Analytics (CSV-mode)")

st.markdown("""
Цей додаток читає дані з CSV (можна завантажити свій файл) і будує:
- графіки відвідуваності;
- розподіл джерел трафіку;
- сегментацію за пристроями;
- кореляційний аналіз ключових метрик.

**Формат CSV (приклад колонок)**:
`date, sessions, users, pageviews, bounce_rate, avg_session_duration, source, medium, device_category`
""")

# --- Sidebar: Upload / sample ---
st.sidebar.header("Джерело даних")
data_source = st.sidebar.radio("Оберіть джерело", ("Завантажити CSV", "Згенерувати приклад CSV"))

@st.cache_data
def generate_sample_csv(n_days=180):
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    sources = ['organic', 'direct', 'referral', 'social', 'email']
    devices = ['desktop', 'mobile', 'tablet']
    rows = []
    np.random.seed(42)
    for d in rng:
        base = 1000 + int(200*np.sin((d.dayofyear/365.0)*2*np.pi))  # seasonality
        for s in sources:
            sessions = max(0, int(np.random.poisson(base * (0.2 if s=='social' else 0.25))))
            users = int(sessions * (0.9 - 0.1*np.random.rand()))
            pageviews = int(sessions * (1.5 + 0.5*np.random.rand()))
            bounce_rate = np.clip(0.3 + 0.2*np.random.rand(), 0, 1)
            avg_session_duration = abs(np.random.normal(120, 30))  # seconds
            device = np.random.choice(devices, p=[0.55, 0.35, 0.10])
            rows.append({
                "date": d.date().isoformat(),
                "sessions": sessions,
                "users": users,
                "pageviews": pageviews,
                "bounce_rate": round(bounce_rate, 3),
                "avg_session_duration": int(avg_session_duration),
                "source": s,
                "medium": "organic" if s=='organic' else ("social" if s=='social' else "referral"),
                "device_category": device
            })
    df = pd.DataFrame(rows)
    # add aggregated daily totals as another view (optional)
    return df

uploaded_df = None
if data_source == "Завантажити CSV":
    uploaded_file = st.sidebar.file_uploader("Завантажте CSV файл (UTF-8)", type=['csv'])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV завантажено")
        except Exception as e:
            st.sidebar.error(f"Не вдалося прочитати CSV: {e}")
else:
    sample_days = st.sidebar.slider("Кількість днів у прикладі", min_value=30, max_value=720, value=180, step=10)
    if st.sidebar.button("Згенерувати прикладні дані"):
        uploaded_df = generate_sample_csv(sample_days)
        csv_buffer = StringIO()
        uploaded_df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button("Завантажити згенерований CSV", data=csv_buffer.getvalue(), file_name="sample_traffic.csv", mime="text/csv")
        st.sidebar.success("Прикладні дані згенеровано")

# If no data yet, offer example generation directly in main
if uploaded_df is None:
    st.info("Завантажте CSV в сайдбарі або згенеруйте прикладні дані")
    st.stop()

# --- Preprocess ---
df = uploaded_df.copy()
# try to parse date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    st.error("CSV має містити колонку `date`. Перейменуйте відповідно та перезавантажте.")
    st.stop()

# standardize lower-case column names for convenience
df.columns = [c.strip() for c in df.columns]

# fill missing expected columns with defaults if absent
for col in ['sessions', 'users', 'pageviews', 'bounce_rate', 'avg_session_duration', 'source', 'device_category']:
    if col not in df.columns:
        if col in ['sessions','users','pageviews','avg_session_duration']:
            df[col] = 0
        elif col == 'bounce_rate':
            df[col] = 0.0
        else:
            df[col] = 'unknown'

# ensure numeric types
for c in ['sessions','users','pageviews','bounce_rate','avg_session_duration']:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# --- Controls ---
st.sidebar.header("Налаштування візуалізацій")
date_range = st.sidebar.date_input("Діапазон дат", [df['date'].min().date(), df['date'].max().date()])
group_by = st.sidebar.selectbox("Групувати по", ['date', 'source', 'device_category'], index=0)
show_sources = st.sidebar.multiselect("Джерела для показу (існуючі)", options=sorted(df['source'].unique()), default=list(sorted(df['source'].unique())))

# filter date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df = df.loc[mask].copy()

st.markdown(f"**Дані з {df['date'].min().date()} по {df['date'].max().date()} — рядків: {len(df)}**")

# --- Traffic over time (daily totals) ---
st.subheader("Відвідуваність у часі")
daily = df.groupby('date').agg({
    'sessions':'sum',
    'users':'sum',
    'pageviews':'sum'
}).reset_index().sort_values('date')

# line chart (Altair)
base = alt.Chart(daily).transform_fold(
    ['sessions','users','pageviews'],
    as_=['metric','value']
).encode(
    x=alt.X('date:T', title='Дата'),
    y=alt.Y('value:Q', title='Кількість'),
    color='metric:N',
    tooltip=['date:T','metric:N','value:Q']
)
line = base.mark_line().interactive()
points = base.mark_circle(size=30).encode(opacity=alt.value(0.6))
st.altair_chart((line + points).properties(height=350), use_container_width=True)

# --- Traffic sources (stacked) ---
st.subheader("Джерела трафіку")
source_agg = df.groupby(['date','source']).agg({'sessions':'sum'}).reset_index()
source_agg = source_agg[source_agg['source'].isin(show_sources)]
if source_agg.empty:
    st.warning("Немає даних для обраних джерел у цьому діапазоні.")
else:
    area = alt.Chart(source_agg).mark_area().encode(
        x='date:T',
        y=alt.Y('sessions:Q', stack='normalize', title='Доля сесій (нормалізовано)'),
        color='source:N',
        tooltip=['date:T','source:N','sessions:Q']
    ).interactive().properties(height=300)
    st.altair_chart(area, use_container_width=True)

    # also show absolute stacked
    area_abs = alt.Chart(source_agg).mark_area().encode(
        x='date:T',
        y=alt.Y('sessions:Q', stack='zero', title='Сесії'),
        color='source:N',
        tooltip=['date:T','source:N','sessions:Q']
    ).interactive().properties(height=300)
    st.caption("Нижче — абсолютні сесії по джерелам")
    st.altair_chart(area_abs, use_container_width=True)

# --- Device segmentation ---
st.subheader("Сегментація за пристроями")
device_agg = df.groupby('device_category').agg({
    'sessions':'sum',
    'users':'sum',
    'pageviews':'sum'
}).reset_index().sort_values('sessions', ascending=False)
st.table(device_agg)
