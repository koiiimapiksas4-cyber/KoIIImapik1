import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import glob

st.set_page_config(layout="wide", page_title="Автоматична аналітика трафіку")

st.title("📊 Аналітика трафіку вебсайту (автоматичне завантаження даних)")

# --- Функція генерації даних ---
@st.cache_data
def generate_sample_csv(n_days=180):
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    sources = ['organic', 'direct', 'referral', 'social', 'email']
    devices = ['desktop', 'mobile', 'tablet']
    rows = []
    np.random.seed(42)
    for d in rng:
        base = 1000 + int(200 * np.sin((d.dayofyear / 365.0) * 2 * np.pi))
        for s in sources:
            sessions = max(0, int(np.random.poisson(base * (0.2 if s == 'social' else 0.25))))
            users = int(sessions * (0.9 - 0.1 * np.random.rand()))
            pageviews = int(sessions * (1.5 + 0.5 * np.random.rand()))
            bounce_rate = np.clip(0.3 + 0.2 * np.random.rand(), 0, 1)
            avg_session_duration = abs(np.random.normal(120, 30))
            device = np.random.choice(devices, p=[0.55, 0.35, 0.10])
            rows.append({
                "date": d.date().isoformat(),
                "sessions": sessions,
                "users": users,
                "pageviews": pageviews,
                "bounce_rate": round(bounce_rate, 3),
                "avg_session_duration": int(avg_session_duration),
                "source": s,
                "medium": "organic" if s == 'organic' else ("social" if s == 'social' else "referral"),
                "device_category": device
            })
    return pd.DataFrame(rows)

# --- Автоматичне завантаження CSV ---
@st.cache_data
def load_data_auto():
    # Шукаємо всі CSV у папці
    csv_files = glob.glob("*.csv")
    if csv_files:
        # Беремо останній за датою створення
        latest_file = max(csv_files, key=os.path.getctime)
        st.success(f"✅ Завантажено CSV-файл: {latest_file}")
        df = pd.read_csv(latest_file)
    else:
        st.warning("⚠️ CSV файлів не знайдено — згенеровано прикладні дані")
        df = generate_sample_csv(180)
    return df

df = load_data_auto()

# --- Попередня обробка ---
df.columns = [c.strip() for c in df.columns]
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    st.error("Файл має містити колонку 'date'")
    st.stop()

for col in ['sessions', 'users', 'pageviews', 'bounce_rate', 'avg_session_duration']:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

for col in ['source', 'device_category']:
    if col not in df.columns:
        df[col] = 'unknown'

# --- Основні показники ---
st.subheader("Загальні показники")
st.metric("📅 Період", f"{df['date'].min().date()} — {df['date'].max().date()}")
st.metric("👥 Унікальні користувачі", f"{df['users'].sum():,}")
st.metric("📈 Сесії", f"{df['sessions'].sum():,}")
st.metric("👁️ Перегляди сторінок", f"{df['pageviews'].sum():,}")

# --- Графік відвідуваності ---
st.subheader("Відвідуваність у часі")
daily = df.groupby('date').agg({
    'sessions': 'sum',
    'users': 'sum',
    'pageviews': 'sum'
}).reset_index()

chart = (
    alt.Chart(daily)
    .transform_fold(['sessions', 'users', 'pageviews'], as_=['metric', 'value'])
    .mark_line(point=True)
    .encode(
        x='date:T',
        y='value:Q',
        color='metric:N',
        tooltip=['date:T', 'metric:N', 'value:Q']
    )
    .interactive()
    .properties(height=350)
)
st.altair_chart(chart, use_container_width=True)

# --- Джерела трафіку ---
st.subheader("Джерела трафіку")
source_agg = df.groupby(['date', 'source']).agg({'sessions': 'sum'}).reset_index()

area = (
    alt.Chart(source_agg)
    .mark_area()
    .encode(
        x='date:T',
        y=alt.Y('sessions:Q', stack='zero', title='Сесії'),
        color='source:N',
        tooltip=['date:T', 'source:N', 'sessions:Q']
    )
    .interactive()
)
st.altair_chart(area, use_container_width=True)

# --- Пристрої ---
st.subheader("Сегментація за пристроями")
device_agg = df.groupby('device_category').agg({
    'sessions': 'sum',
    'users': 'sum',
    'pageviews': 'sum'
}).reset_index()

st.table(device_agg)

bar = (
    alt.Chart(device_agg)
    .mark_bar()
    .encode(
        x='sessions:Q',
        y=alt.Y('device_category:N', sort='-x'),
        color='device_category:N',
        tooltip=['device_category', 'sessions', 'users', 'pageviews']
    )
)
st.altair_chart(bar, use_container_width=True)

# --- Кореляційний аналіз без matplotlib ---
st.subheader("Кореляційний аналіз показників (без matplotlib)")

num_cols = ['sessions', 'users', 'pageviews', 'bounce_rate', 'avg_session_duration']
corr = df[num_cols].corr().round(3)

st.dataframe(corr)

corr_long = (
    corr.reset_index()
        .melt(id_vars='index', var_name='var2', value_name='corr')
        .rename(columns={'index': 'var1'})
)

heat = (
    alt.Chart(corr_long)
    .mark_rect()
    .encode(
        x='var1:N',
        y='var2:N',
        color=alt.Color('corr:Q', scale=alt.Scale(domain=[-1,1], scheme='redyellowblue')),
        tooltip=['var1','var2','corr']
    )
)

text = (
    alt.Chart(corr_long)
    .mark_text(size=12)
    .encode(
        x='var1:N',
        y='var2:N',
        text='corr:Q',
        color=alt.condition(alt.datum.corr > 0.5, alt.value('black'), alt.value('black'))
    )
)

st.altair_chart(heat + text, use_container_width=True)

st.caption("""
**Інтерпретація:**  
- corr близько 1 → сильна позитивна залежність  
- corr близько -1 → сильна негативна залежність  
- corr близько 0 → лінійної залежності немає
""")
