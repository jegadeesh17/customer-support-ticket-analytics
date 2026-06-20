"""Generate EDA visualizations and save to docs/eda/."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_tickets
from src.paths import get_eda_dir
from src.preprocessor import standardize_columns

sns.set_theme(style='whitegrid', palette='muted')


def _save(fig, name):
    path = os.path.join(get_eda_dir(), name)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def generate_shared_eda(df):
    df = standardize_columns(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    df['priority'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Ticket Priority Distribution')
    ax.set_xlabel('Priority')
    ax.set_ylabel('Count')
    _save(fig, '01_priority_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    df['customer_satisfaction_score'].dropna().astype(int).value_counts().sort_index().plot(
        kind='bar', ax=ax, color='seagreen'
    )
    ax.set_title('Customer Satisfaction Score Distribution')
    ax.set_xlabel('Score')
    _save(fig, '02_satisfaction_distribution.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['resolution_time_hours'], kde=True, bins=40, ax=ax, color='teal')
    ax.set_title('Resolution Time Distribution (Hours)')
    _save(fig, '03_resolution_time_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    df['channel'].value_counts().plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Tickets by Channel')
    ax.tick_params(axis='x', rotation=45)
    _save(fig, '04_channel_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    df['region'].value_counts().plot(kind='bar', ax=ax, color='mediumpurple')
    ax.set_title('Tickets by Region')
    ax.tick_params(axis='x', rotation=45)
    _save(fig, '05_region_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    df['subscription_type'].value_counts().plot(kind='bar', ax=ax, color='goldenrod')
    ax.set_title('Subscription Type Distribution')
    _save(fig, '06_subscription_type_distribution.png')

    numeric = df.select_dtypes(include='number')
    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric.corr(), annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Numeric Feature Correlation Heatmap')
        _save(fig, '07_correlation_heatmap.png')

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        missing.plot(kind='barh', ax=ax, color='gray')
        ax.set_title('Missing Values by Column')
        _save(fig, '08_missing_values.png')
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No missing values detected', ha='center', va='center', fontsize=14)
        ax.axis('off')
        _save(fig, '08_missing_values.png')

    sla = df.groupby('priority')['sla_breached'].apply(
        lambda s: (s.astype(str).str.lower() == 'yes').mean()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sla.plot(kind='bar', ax=ax, color='indianred')
    ax.set_title('SLA Breach Rate by Priority')
    ax.set_ylabel('Breach Rate')
    _save(fig, '09_sla_breach_by_priority.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='priority', y='resolution_time_hours', showfliers=False, ax=ax)
    ax.set_title('Resolution Time by Priority')
    _save(fig, '10_resolution_time_by_priority.png')


def generate_classification_eda(df):
    df = standardize_columns(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    df['desc_length'] = df['issue_description'].fillna('').astype(str).str.len()
    sns.boxplot(data=df, x='priority', y='desc_length', showfliers=False, ax=ax)
    ax.set_title('Description Length vs Priority')
    _save(fig, 'cls_01_desc_length_by_priority.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='priority', y='first_response_time_hours', showfliers=False, ax=ax)
    ax.set_title('First Response Time vs Priority')
    _save(fig, 'cls_02_response_time_by_priority.png')


def generate_regression_eda(df):
    df = standardize_columns(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='category', y='resolution_time_hours', showfliers=False, ax=ax)
    ax.set_title('Resolution Time by Category')
    ax.tick_params(axis='x', rotation=90)
    _save(fig, 'reg_01_resolution_by_category.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x='first_response_time_hours',
        y='resolution_time_hours',
        hue='sla_breached',
        alpha=0.4,
        ax=ax,
    )
    ax.set_title('Resolution Time vs First Response (colored by SLA breach)')
    _save(fig, 'reg_02_sla_vs_resolution_scatter.png')


def generate_satisfaction_eda(df):
    df = standardize_columns(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='channel', y='customer_satisfaction_score', ax=ax)
    ax.set_title('Satisfaction by Channel')
    ax.tick_params(axis='x', rotation=45)
    _save(fig, 'sat_01_satisfaction_by_channel.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='customer_segment', y='customer_satisfaction_score', ax=ax)
    ax.set_title('Satisfaction by Customer Segment')
    ax.tick_params(axis='x', rotation=45)
    _save(fig, 'sat_02_satisfaction_by_segment.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='subscription_type', y='customer_satisfaction_score', ax=ax)
    ax.set_title('Satisfaction by Subscription Type')
    _save(fig, 'sat_03_satisfaction_by_subscription.png')


def main():
    df = load_tickets(use_db=False)
    if df.empty:
        print('No data for EDA.')
        return
    generate_shared_eda(df)
    generate_classification_eda(df)
    generate_regression_eda(df)
    generate_satisfaction_eda(df)
    print('EDA generation complete.')


if __name__ == '__main__':
    main()
