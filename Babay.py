import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_csv(path):
    """Завантажує CSV. Автоматично парсить date."""
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError('CSV повинен містити стовпчик "date"')
    # Переконаємося у наявності ключових полів
    expected = ['sessions','users','pageviews','traffic_source','device_category']
    for c in expected:
        if c not in df.columns:
            print(f'У CSV відсутній стовпець: {c}. Значення будуть заповнені NaN')
            df[c] = df.get(c, np.nan)
    return df
def load_ga4(service_account_json, property_id, start_date=None, end_date=None):
    """Завантажує дані з Google Analytics Data API (GA4).
    Потрібно встановити: pip install google-analytics-data
    Перед використанням переконайтеся, що service account має доступ до property.
    Ця функція робить базовий запит: sessions, users, pageviews, deviceCategory, sessionSource
    """
    try:
        from google.analytics.data import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import DateRange, Metric, Dimension, RunReportRequest
    except Exception as e:
        raise RuntimeError('Не знайдено google-analytics-data. Встановіть: pip install google-analytics-data')
    client = BetaAnalyticsDataClient.from_service_account_file(service_account_json)
    if start_date is None:
        start_date = '30daysAgo'
    if end_date is None:
        end_date = 'today'
    request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[Dimension(name='date'), Dimension(name='sessionSource'), Dimension(name='deviceCategory')],
        metrics=[Metric(name='sessions'), Metric(name='activeUsers'), Metric(name='screenPageViews'), Metric(name='averageSessionDuration')],
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        limit=100000,
    )
    response = client.run_report(request)
    rows = []
    for row in response.rows:
        dims = [d.value for d in row.dimension_values]
        mets = [m.value for m in row.metric_values]
        rows.append(dims + mets)

    cols = [d.name for d in request.dimensions] + [m.name for m in request.metrics]
    df = pd.DataFrame(rows, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'sessionSource':'traffic_source','deviceCategory':'device_category','sessions':'sessions','activeUsers':'users','screenPageViews':'pageviews','averageSessionDuration':'avg_session_duration'})
    for c in ['sessions','users','pageviews','avg_session_duration']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
def preprocess(df):
    """Агрегація за датою, заповнення пропусків, базові метрики."""
    df = df.copy()
    if 'date' not in df.columns:
        raise ValueError('Дані повинні містити стовпець date')
    for num in ['sessions','users','pageviews','bounce_rate','avg_session_duration']:
        if num not in df.columns:
            df[num] = np.nan
    daily = df.groupby('date').agg({
        'sessions':'sum',
        'users':'sum',
        'pageviews':'sum',
        'bounce_rate':'mean',
        'avg_session_duration':'mean'
    }).reset_index()
    idx = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(idx).rename_axis('date').reset_index()
    daily[['sessions','users','pageviews']] = daily[['sessions','users','pageviews']].fillna(0)
    return daily
def plot_time_series(daily, outdir):
    """Графік відвідуваності: sessions, users, pageviews"""
    plt.figure(figsize=(12,5))
    plt.plot(daily['date'], daily['sessions'], label='sessions')
    plt.plot(daily['date'], daily['users'], label='users')
    plt.plot(daily['date'], daily['pageviews'], label='pageviews')
    plt.title('Відвідуваність — щоденний показник')
    plt.xlabel('Дата')
    plt.ylabel('Кількість')
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, 'time_series_visits.png')
    plt.savefig(p)
    plt.close()
    print('Збережено:', p)
def plot_traffic_sources(df, outdir, top_n=10):
    """Барплот по джерелам трафіку (агреговано)."""
    src = df.groupby('traffic_source')['sessions'].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    plt.bar(src.index.astype(str), src.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Топ {top_n} джерел трафіку (за сесіями)')
    plt.ylabel('sessions')
    plt.tight_layout()
    p = os.path.join(outdir, 'traffic_sources.png')
    plt.savefig(p)
    plt.close()
    print('Збережено:', p)
def plot_device_segmentation(df, outdir):
    """Сегментація за device_category: pie та bar"""
    dev = df.groupby('device_category')['sessions'].sum().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    plt.bar(dev.index.astype(str), dev.values)
    plt.title('Сегментація за пристроями (sessions)')
    plt.tight_layout()
    p1 = os.path.join(outdir, 'device_segmentation_bar.png')
    plt.savefig(p1)
    plt.close()
    plt.figure(figsize=(6,6))
    plt.pie(dev.values, labels=dev.index.astype(str), autopct='%1.1f%%')
    plt.title('Сегментація за пристроями (sessions)')
    p2 = os.path.join(outdir, 'device_segmentation_pie.png')
    plt.savefig(p2)
    plt.close()
    print('Збережено:', p1, p2)
def correlation_analysis(daily, outdir):
    """Рахує кореляцію між числовими показниками і будує heatmap."""
    nums = daily.select_dtypes(include=[np.number])
    if nums.shape[1] < 2:
        print('Недостатньо числових полів для кореляції')
        return None
    corr = nums.corr(method='pearson')
    corr_path = os.path.join(outdir, 'correlation_matrix.csv')
    corr.to_csv(corr_path)
    print('Збережено матрицю кореляції:', corr_path)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, interpolation='nearest', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            text = ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.title('Матриця кореляцій (Pearson)')
    plt.tight_layout()
    heatmap_path = os.path.join(outdir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print('Збережено:', heatmap_path)
    return corr
def segment_by_device(df):
    """Повертає агреговані показники по типам пристроїв."""
    seg = df.groupby('device_category').agg({
        'sessions':'sum',
        'users':'sum',
        'pageviews':'sum'
    }).reset_index()
    return seg
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    if args.csv:
        df_raw = load_csv(args.csv)
    elif args.ga_key and args.property_id:
        df_raw = load_ga4(args.ga_key, args.property_id, start_date=args.start_date, end_date=args.end_date)
    else:
        raise ValueError('Вкажіть або --csv, або --ga_key та --property_id')

    daily = preprocess(df_raw)
    plot_time_series(daily, args.outdir)
    if 'traffic_source' in df_raw.columns:
        plot_traffic_sources(df_raw, args.outdir)
    if 'device_category' in df_raw.columns:
        plot_device_segmentation(df_raw, args.outdir)
    corr = correlation_analysis(daily, args.outdir)
    if corr is not None:
        print('\nНайсильніші пари кореляцій (за абсолютним значенням):')
        corr_abs = corr.abs()
        np.fill_diagonal(corr_abs.values, 0)
        flat = corr_abs.unstack().sort_values(ascending=False).drop_duplicates()
        print(flat.head(10))
    if 'device_category' in df_raw.columns:
        seg = segment_by_device(df_raw)
        seg_path = os.path.join(args.outdir, 'device_segmentation.csv')
        seg.to_csv(seg_path, index=False)
        print('Збережено сегментацію:', seg_path)
    print('\nГотово.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Аналітика трафіку вебсайту')
    parser.add_argument('--csv', help='Шлях до CSV файлу')
    parser.add_argument('--ga_key', help='Шлях до service account JSON для GA4')
    parser.add_argument('--property_id', help='GA4 property id (число)')
    parser.add_argument('--start_date', help="Початкова дата для GA4 (YYYY-MM-DD або '30daysAgo')", default=None)
    parser.add_argument('--end_date', help="Кінцева дата для GA4 (YYYY-MM-DD або 'today')", default=None)
    parser.add_argument('--outdir', help='Папка для результатів', default='results')
    args = parser.parse_args()
    main(args)

