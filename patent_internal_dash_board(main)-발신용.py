#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install dash --trusted-host pypi.org --trusted-host files.pythonhosted.org')


# In[ ]:


get_ipython().system('pip install jupyter-dash --trusted-host pypi.org --trusted-host files.pythonhosted.org')


# In[ ]:


get_ipython().system('pip install geopandas --trusted-host pypi.org --trusted-host files.pythonhosted.org')


# In[ ]:


get_ipython().system('pip install --upgrade xlrd')


# In[ ]:


get_ipython().system('pip install jupyter-dash')


# In[ ]:


get_ipython().system('pip install --upgrade openpyxl pandas')


# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import skew
import dash
from jupyter_dash import JupyterDash
import plotly.express as px
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from matplotlib import pyplot as plt
import openpyxl
from dash.dependencies import Input, Output, State
import geopandas as gpd
from dash import html
from dash import dcc
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
import os


# In[2]:


cwd=os.getcwd()


# In[3]:


join_df_for_treemap_animation=pd.read_csv(cwd+'/data/join_df_for_treemap_animation.csv')
join_df_for_wipo_treemap_animation=pd.read_csv(cwd+'/data/join_df_for_wipo_treemap_animation.csv')
scheme_for_search=pd.read_csv(cwd+'/data/scheme_for_search.csv')


# # Treemap 시각화

# In[4]:


# 색상 매핑 설정 (섹션별 파스텔톤 색상)
base_colors = {
    'A': 'rgb(173, 216, 230)',  # pastel blue
    'B': 'rgb(224, 255, 255)',  # pastel light cyan
    'C': 'rgb(255, 182, 193)',  # pastel pink
    'D': 'rgb(255, 223, 186)',  # pastel peach
    'E': 'rgb(255, 255, 204)',  # pastel yellow
    'F': 'rgb(204, 204, 255)',  # pastel purple
    'G': 'rgb(255, 204, 229)',  # pastel pink purple
    'H': 'rgb(255, 228, 225)',  # pastel misty rose
    'Y': 'rgb(211, 211, 211)',  # pastel gray
    'Z': 'rgb(144, 238, 144)',   # pastel green
    'wipo' : 'rgb(200, 200, 200)' 
}

# Dash 애플리케이션 생성
app = dash.Dash(__name__)

# 출원년도 리스트 생성 (두 데이터프레임의 공통된 년도 사용)
years = list(set(join_df_for_treemap_animation['출원년도'].unique()).intersection(set(join_df_for_wipo_treemap_animation['출원년도'].unique())))
years.sort()

# 슬라이더 마크 설정 (모든 년도에 대해 표시)
slider_marks = {str(year): {'label': str(year), 'style': {'color': '#7fafdf' if year % 2 == 0 else '#b0b0b0'}} for year in years}

def adjust_color(color, level):
    r, g, b = [int(c) for c in color[4:-1].split(',')]
    factor = 1 - (level - 1) * 0.1  # 레벨에 따라 색상 명도 조절
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f'rgb({r}, {g}, {b})'

def split_title(title, max_length=40, initial_offset=0):
    words = title.split(' ')
    lines = []
    current_line = []
    current_length = initial_offset
    for word in words:
        if current_length + len(word) <= max_length:
            current_line.append(word)
            current_length += len(word) + 1  # 단어 사이의 공백을 포함하여 계산
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word) + 1
    if current_line:
        lines.append(' '.join(current_line))
    return '<br>'.join(lines)

def generate_figure1(df, year, search_term=None):
    df_seperated = df[(df['출원년도'] == year) & (df['누적건수_total'] >= 0)].copy()
    df_seperated = df_seperated[(df_seperated['누적건수_total'] > 0) & (df_seperated['level'] <= 5)]
    
    if df_seperated.empty:
        return go.Figure()

    df_seperated['label'] = df_seperated.apply(
        lambda row: f"{row['label']} ({row['누적건수_total']:,}건)" if row['level'] in [1, 2] else row['label'],
        axis=1
    )
    
    # NaN 값 처리
    df_seperated['parent_cpc'] = df_seperated['parent_cpc'].fillna('')
    df_seperated['label'] = df_seperated['label'].fillna('')
    df_seperated['title'] = df_seperated['title'].apply(lambda x: split_title(x, 40, initial_offset=len('Title : ')))
    df_seperated['font_size'] = np.where(df_seperated['level'] == 4, np.maximum(np.log(df_seperated['누적건수_total']) * 5, 10), 12)
    
    df_seperated['소분류'] = df_seperated['소분류'].apply(lambda x: '' if pd.isna(x) else x)
    df_seperated['소분류'] = df_seperated['소분류'].apply(lambda x: split_title(x, 40, initial_offset=len('포함될 수 있는 분야 : ')))
    
    def generate_hovertext(row):
        base_text = f"<b>{row['label']}</b><br><br>건수 : {row['누적건수_total']:,}건<br><b>Title : {row['title']}</b>"
        if row['소분류']:
            return base_text + f"<br><b>포함될 수 있는 분야 : <span style='color:blue;'>{row['소분류']}</span></b><extra></extra>"
        else:
            return base_text + "<extra></extra>"
    
    df_seperated['hovertext'] = df_seperated.apply(generate_hovertext, axis=1)
    
    colors = [adjust_color(base_colors.get(section, 'rgb(211, 211, 211)'), level) for section, level in zip(df_seperated['section'], df_seperated['level'])]

    # Highlight based on search term in scheme_for_search dataframe or CPC code
    if search_term:
        highlight_color = 'rgb(255, 0, 0)'  # 강조 색상 (빨간색)
        search_term_lower = search_term.lower()

        # 검색어가 scheme_for_search에 포함된 경우 CPC 코드를 찾음
        matching_cpcs = scheme_for_search[
            scheme_for_search['eng_title_extended'].str.contains(search_term_lower, case=False, na=False) |
            scheme_for_search['kor_title_extended'].str.contains(search_term_lower, case=False, na=False)
        ]['cpc_for_search'].unique()

        # 검색어가 직접 CPC 코드인 경우 해당 코드 추가
        if search_term in df_seperated['cpc'].values:
            matching_cpcs = np.append(matching_cpcs, search_term)

        # 색상 하이라이트 로직
        colors = [
            highlight_color if cpc in matching_cpcs else color
            for cpc, color in zip(df_seperated['cpc'], colors)
        ]
    
    cumulative_count = df_seperated[df_seperated['level'] == 1]['누적건수_total'].sum()
    max_cumulative = df[df['level'] == 1]['누적건수_total'].max()
    scale_factor = np.log1p(cumulative_count) / np.log1p(max_cumulative) 
    scale_factor = max(scale_factor, 0.05)
    
    trace = go.Treemap(
        labels=df_seperated['label'],
        values=df_seperated['누적건수_total'],
        ids=df_seperated['cpc'],
        parents=df_seperated['parent_cpc'],
        branchvalues='total',
        texttemplate='<b>%{label}</b>',
        hovertemplate=df_seperated['hovertext'],
        pathbar_textfont_size=15,
        customdata=df_seperated[['title', '소분류']],
        marker=dict(
            colors=colors,
            line=dict(color='black', width=1)
        ),
        insidetextfont=dict(size=df_seperated['font_size'], family="Arial, sans-serif", color="black"),
        textposition='middle center',
        domain=dict(x=[0.5 - scale_factor / 2, 0.5 + scale_factor / 2], y=[0.5 - scale_factor / 2, 0.5 + scale_factor / 2])
    )
    
    fig = go.Figure(data=[trace])
    fig.update_layout(
        paper_bgcolor='rgb(235, 235, 235)'
    )
    
    return fig

def generate_figure2(df, year, search_term=None):
    df_seperated = df[(df['출원년도'] == year) & (df['누적건수_total'] >= 0)].copy()
    df_seperated = df_seperated[(df_seperated['누적건수_total'] > 0) & (df_seperated['level'] <= 4)]
    
    if df_seperated.empty:
        return go.Figure()

    df_seperated['label'] = df_seperated.apply(
        lambda row: f"{row['label']} ({row['누적건수_total']:,}건)" if row['level'] in [1] else row['label'],
        axis=1
    )
    
    # NaN 값 처리
    df_seperated['parent_cpc'] = df_seperated['parent_cpc'].fillna('')
    df_seperated['label'] = df_seperated['label'].fillna('')
    df_seperated['title'] = df_seperated['title'].apply(lambda x: split_title(x, 40, initial_offset=len('Title : ')))
    df_seperated['font_size'] = np.where(df_seperated['level'] == 4, np.maximum(np.log(df_seperated['누적건수_total']) * 5, 10), 12)
    
    df_seperated['소분류'] = df_seperated['소분류'].apply(lambda x: '' if pd.isna(x) else x)
    df_seperated['소분류'] = df_seperated['소분류'].apply(lambda x: split_title(x, 40, initial_offset=len('포함될 수 있는 분야 : ')))
    
    def generate_hovertext(row):
        base_text = f"<b>{row['label']}</b><br><br>건수 : {row['누적건수_total']:,}건<br><b>Title : {row['title']}</b>"
        if row['소분류']:
            return base_text + f"<br><b>포함될 수 있는 분야 : <span style='color:blue;'>{row['소분류']}</span></b><extra></extra>"
        else:
            return base_text + "<extra></extra>"
    
    df_seperated['hovertext'] = df_seperated.apply(generate_hovertext, axis=1)
    
    colors = [adjust_color(base_colors.get(section, 'rgb(211, 211, 211)'), level) for section, level in zip(df_seperated['section'], df_seperated['level'])]

    # Highlight based on search term in scheme_for_search dataframe or CPC code
    if search_term:
        highlight_color = 'rgb(255, 0, 0)'  # 강조 색상 (빨간색)
        search_term_lower = search_term.lower()

        # 검색어가 scheme_for_search에 포함된 경우 CPC 코드를 찾음
        matching_cpcs = scheme_for_search[
            scheme_for_search['eng_title_extended'].str.contains(search_term_lower, case=False, na=False) |
            scheme_for_search['kor_title_extended'].str.contains(search_term_lower, case=False, na=False)
        ]['cpc_for_search'].unique()

        # 검색어가 직접 CPC 코드인 경우 해당 코드 추가
        if search_term in df_seperated['cpc'].values:
            matching_cpcs = np.append(matching_cpcs, search_term)

        # 색상 하이라이트 로직
        colors = [
            highlight_color if cpc in matching_cpcs else color
            for cpc, color in zip(df_seperated['cpc'], colors)
        ]
    
    cumulative_count = df_seperated[df_seperated['level'] == 1]['누적건수_total'].sum()
    max_cumulative = df[df['level'] == 1]['누적건수_total'].max()
    scale_factor = np.log1p(cumulative_count) / np.log1p(max_cumulative) 
    scale_factor = max(scale_factor, 0.05)
    
    trace = go.Treemap(
        labels=df_seperated['label'],
        values=df_seperated['누적건수_total'],
        ids=df_seperated['cpc'],
        parents=df_seperated['parent_cpc'],
        branchvalues='total',
        texttemplate='<b>%{label}</b>',
        hovertemplate=df_seperated['hovertext'],
        pathbar_textfont_size=15,
        customdata=df_seperated[['title', '소분류']],
        marker=dict(
            colors=colors,
            line=dict(color='black', width=1)
        ),
        insidetextfont=dict(size=df_seperated['font_size'], family="Arial, sans-serif", color="black"),
        textposition='middle center',
        domain=dict(x=[0.5 - scale_factor / 2, 0.5 + scale_factor / 2], y=[0.5 - scale_factor / 2, 0.5 + scale_factor / 2])
    )
    
    fig = go.Figure(data=[trace])
    fig.update_layout(
        paper_bgcolor='rgb(235, 235, 235)'
    )
    
    return fig

# 슬라이더 마크 설정 (모든 년도에 대해 표시하되, 라벨은 2년마다 표시)
slider_marks = {
    str(year): {'label': str(year) if year % 2 == 0 else '', 'style': {'color': '#7fafdf' if year % 2 == 0 else '#b0b0b0'}}
    for year in years
}

app.layout = html.Div([
    html.H1("Treemap Dashboard", style={'textAlign': 'center'}),
    
    # 라디오 버튼, 검색창, 검색 버튼을 하나의 행에 배치
    html.Div([
        dcc.RadioItems(
            id='treemap-radio',
            options=[
                {'label': 'CPC Treemap', 'value': 'CPC'},
                {'label': 'WIPO 35대분류 Treemap', 'value': 'WIPO'}
            ],
            value='CPC',  # 기본값으로 'CPC Treemap' 선택
            inline=True,
            style={'display': 'inline-block', 'margin-right': '20px'}
        ),
        dcc.Input(
            id='label-search',
            type='text',
            placeholder='Search CPC codes or Title kewords',
            style={'width': '300px', 'display': 'inline-block', 'margin-right': '12px'}
        ),
        html.Button(id='search-button', n_clicks=0, children='🔍', style={'display': 'inline-block', 'font-size': '12px'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'padding': '12px'}),

    # 단일 트리맵 그래프
    dcc.Graph(id='treemap-graph', style={'height': '60vh', 'width': '90%', 'margin': 'auto'}),
    
    html.Div([
        html.Div([
            dcc.Slider(
                id='year-slider',
                min=years[0],
                max=years[-1],
                value=years[0],  # 슬라이더 초기 값을 데이터의 첫 번째 해로 설정
                marks=slider_marks,  # 모든 년도에 대해 마크를 설정
                step=1,  # 모든 년도에 대해 접근 가능하도록 설정
                tooltip={"always_visible": True, "placement": "bottom"}  # 슬라이더 도구 팁 항상 표시
            )
        ], style={'width': '80%', 'display': 'inline-block'}),
        html.Button(id='play-button', n_clicks=0, children='▶️', style={'display': 'inline-block', 'margin-left': '10px'}),
        html.Button(id='pause-button', n_clicks=0, children='❚❚', style={'display': 'inline-block', 'margin-left': '5px'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'padding': '10px'}),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1초 간격
        n_intervals=0,
        disabled=True  # 처음에는 비활성화
    )
])

@app.callback(
    Output('interval-component', 'disabled'),
    [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_animation(play_clicks, pause_clicks, is_disabled):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_disabled
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'play-button':
        return False
    elif button_id == 'pause-button':
        return True

    return is_disabled

@app.callback(
    Output('year-slider', 'value'),
    [Input('interval-component', 'n_intervals')],
    [State('year-slider', 'value')]
)
def update_slider(n, current_value):
    if n == 0:
        return current_value  # No change if it's the initial state
    
    next_year = current_value + 1
    if next_year > years[-1]:
        next_year = years[0]
    return next_year

@app.callback(
    Output('treemap-graph', 'figure'),
    [Input('year-slider', 'value'), Input('search-button', 'n_clicks'), Input('treemap-radio', 'value')],
    [State('label-search', 'value')]
)
def update_figures(year, n_clicks, selected_treemap, search_term):
    if n_clicks == 0:
        search_term = None
    
    if selected_treemap == 'CPC':
        # Display CPC Treemap
        fig = generate_figure1(join_df_for_treemap_animation, year, search_term)
    else:
        # Display WIPO Treemap
        fig = generate_figure2(join_df_for_wipo_treemap_animation, year, search_term)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




