"""
Спрощена версія аналітики трафіку без matplotlib.
Використовує plotly.express для графіків.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import argparse

def load_csv(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess(df):
    daily = df.groupby('date').agg({
        'sessions': 'sum',
        'users': 'sum',
        'pageviews': 'sum'
    }).reset_index()
    return daily

def plot_time_series(daily, outdir):
    fig = px.line(
        daily,
        x='date',
        y=['sessions', 'users', 'pageviews'],
        title='Відвідуваність — щоденний показник'
    )
    fig.write_html(os.path.join(outdir, "time_series.html"))

def plot_traffic_sources(df, outdir):
    src = df.groupby('traffic_source')['sessions'].sum().reset_index()
    fig = px.bar(
        src.sort_values('sessions', ascending=False),
        x='traffic_source',
        y='sessions',
        title='Джерела трафіку'
    )
    fig.write_html(os.path.join(outdir, "traffic_sources.html"))

def plot_device_segmentation(df, outdir):
    dev = df.groupby('device_category')['sessions'].sum().reset_index()
    fig = px.pie(
        dev,
        names='device_category',
        values='sessions',
        title='Сегментація за пристроями'
    )
    fig.write_html(os.path.join(outdir, "device_segmentation.html"))

def correlation_analysis(df, outdir):
    nums = df[['sessions', 'users', 'pageviews']]
    corr = nums.corr()
    fig = px.imshow(corr, text_auto=True, title='Матриця кореляцій (Plotly)')
    fig.write_html(os.path.join(outdir, "correlation_heatmap.html"))

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.csv)
    daily = preprocess(df)
    plot_time_series(daily, args.outdir)
    plot_traffic_sources(df, args.outdir)
    plot_device_segmentation(df, args.outdir)
    correlation_analysis(daily, args.outdir)
    print("✅ Усі результати збережено в HTML форматі у папці:", args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Аналітика трафіку без matplotlib")
    parser.add_argument('--csv', required=True, help='Шлях до CSV файлу')
    parser.add_argument('--outdir', default='results', help='Папка для результатів')
    args = parser.parse_args()
    main(args)
