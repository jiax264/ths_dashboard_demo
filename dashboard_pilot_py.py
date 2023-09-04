#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dash_table.Format import Format, Scheme, Sign, Symbol
from dash import no_update
from dash.exceptions import PreventUpdate


# In[2]:


def plot_accident_heatmap(df, start_date, end_date, day=None):
    df = df.copy()
    
    df['date'] = pd.to_datetime(df['date'])

    if end_date < start_date:
        start_date, end_date = end_date, start_date

    if day:
        df = df[df['day_of_week'] == day]

    no_midnight_df = df[df['time'] != datetime.time(0, 0)]

    no_midnight_df = no_midnight_df[
        (no_midnight_df['date'] >= start_date) & 
        (no_midnight_df['date'] <= end_date)
    ]
    
    grouped_df = no_midnight_df.groupby(['roadway_name', 'time_bucket']).size().reset_index(name='accident_count')
    grouped_df = grouped_df.sort_values(by='accident_count', ascending=False)
    
    total_accidents_per_road = grouped_df.groupby('roadway_name')['accident_count'].sum()
    top_15_roads = total_accidents_per_road.nlargest(15).index.tolist()
    filtered_df = grouped_df[grouped_df['roadway_name'].isin(top_15_roads)]

    heatmap_data = filtered_df.pivot(index='roadway_name', columns='time_bucket', values='accident_count').fillna(0)
    heatmap_data = heatmap_data[['0-3', '4-7', '8-11', '12-15', '16-19', '20-23']]
    heatmap_data = heatmap_data.reindex(total_accidents_per_road.nlargest(15).index)

    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="Reds",
        labels=dict(color="Accident Count"),
        title="Accident Counts by Roadway Name and Time"
    )

    return fig


# In[3]:


def plot_accidents_on_map(df, start_date, end_date, day_of_week=None, roadway_name=None):
    df = df.copy()
    time_bucket_colors = {
        '0-3': '#17BECF',
        '4-7': '#FE00FA',
        '8-11': '#FB0D0D',
        '12-15': '#6C7C32',
        '16-19': '#3283FE',
        '20-23': '#750D86'
    }

    df['date'] = pd.to_datetime(df['date'])
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    if day_of_week:
        df = df[df['day_of_week'] == day_of_week]

    no_midnight_df = df[df['time'] != datetime.time(0, 0)]
    no_midnight_df = no_midnight_df[(no_midnight_df['date'] >= start_date) & (no_midnight_df['date'] <= end_date)]

    if roadway_name:
        no_midnight_df = no_midnight_df[no_midnight_df['roadway_name'] == roadway_name]

    # Vanderbilt Stevenson Coordinates to Center Map
    center_lat = 36.14347
    center_lon = -86.80261
    if roadway_name and not no_midnight_df.empty:
        center_lat = no_midnight_df['latitude'].median()
        center_lon = no_midnight_df['longitude'].median()

    zoom_level = 13

    traces = []
    for time_bucket, color in time_bucket_colors.items():
        subset = no_midnight_df[no_midnight_df['time_bucket'] == time_bucket]

        # Create a visible trace if there's data for the current time bucket
        if not subset.empty:
            trace = go.Scattermapbox(
                lon=subset['longitude'],
                lat=subset['latitude'],
                mode='markers',
                marker=dict(color=color, size=7),
                name=time_bucket,
                hovertext=subset['roadway_name'] + "<br>" + subset['hour'].astype(str) + "H",
                legendgroup=time_bucket
            )
        # Otherwise, create a dummy trace to ensure the time bucket is shown in the legend
        else:
            trace = go.Scattermapbox(
                lon=[None],
                lat=[None],
                mode='markers',
                marker=dict(color=color, size=7),
                name=time_bucket,
                showlegend=True,
                legendgroup=time_bucket
            )
        traces.append(trace)

    layout = go.Layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
            style="open-street-map"
        ),
        title='Scatter Map of Accidents',
        margin={"r": 0, "t": 50, "l": 0, "b": 10}
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


# In[4]:


filepath_q4 = os.path.join("..", "01_data", "2022_Q4_Traffic_Crashes.csv")
df_q4 = pd.read_csv(filepath_q4)

filepath_q3 = os.path.join("..", "01_data", "2022_Q3_Traffic_Crashes.csv")
df_q3 = pd.read_csv(filepath_q3)

filepath_q2 = os.path.join("..", "01_data", "2022_Q2_Traffic_Crashes.csv")
df_q2 = pd.read_csv(filepath_q2)

filepath_q1 = os.path.join("..", "01_data", "2022_Q1_Traffic_Crashes.csv")
df_q1 = pd.read_csv(filepath_q1)

df = pd.concat([df_q1, df_q2, df_q3, df_q4], ignore_index=True)

columns_to_keep = [
    "collision_date", 
    "vehicles_involved", 
    "number_injured", 
    "number_dead", 
    "latitude", 
    "longitude", 
    "roadway_name", 
    "intersection_indicator", 
    "intersect_type", 
    "relation_to_junction", 
    "weather_condition(s)", 
    "manner_of_crash", 
    "pedestrian", 
    "bicycle", 
    "scooter", 
    "hitrun"
]

df = df[columns_to_keep]

df['collision_date'] = pd.to_datetime(df['collision_date'], format='%m/%d/%Y %I:%M:%S %p')

df['date'] = df['collision_date'].dt.date
df['day_of_week'] = df['collision_date'].dt.dayofweek
df['time'] = df['collision_date'].dt.time

day_map = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}
df['day_of_week'] = df['day_of_week'].map(day_map)

df['hour'] = df['collision_date'].dt.hour

bins = [-1, 3, 7, 11, 15, 19, 23]  # Defining bins with -1 as the lower edge to ensure 0 is inclusive in the first bin
labels = ['0-3', '4-7', '8-11', '12-15', '16-19', '20-23']

df['time_bucket'] = pd.cut(df['hour'], bins=bins, labels=labels)

latest_datetime = df['collision_date'].max()
latest_date = latest_datetime.to_pydatetime()
earliest_datetime = df['collision_date'].min()
earliest_date = earliest_datetime.to_pydatetime()


# In[5]:


external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets, 
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}] 
) 

home_page_layout = dbc.Container([
    
    dbc.Row(
        dbc.Col(html.H1("TN Highway Safety VUPD Dashboard", 
                        className='text-center mb-4 mt-3'), width=10),
    justify='around'),
    
    dbc.Row([
        dbc.Col(
            [
                html.Div([  
                    dcc.DatePickerRange(
                        id='date-range-picker', 
                        clearable=False,
                        number_of_months_shown=1,
                        min_date_allowed=earliest_date,
                        max_date_allowed=latest_date,
                        display_format='MMM Do, YY',
                        month_format='MMMM, YYYY',
                        initial_visible_month=earliest_date,
                        minimum_nights=6,
                        persistence=True,
                        persisted_props=['start_date', 'end_date'],
                        persistence_type='session',
                        start_date=earliest_date,
                        end_date=latest_date,
                        className='mb-2'
                    ),
                ]),
                html.Div([  
                    html.Label("Select Date Range", style={"marginTop": "1px"})
                ])
            ],
            width=4
        ),
        dbc.Col(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id='day-of-week-input',
                            options=[{'label': i, 'value': i} for i in df['day_of_week'].unique()],
                            clearable=True
                        ), 
                        html.Div(id='hidden-div-day-of-week', style={'display':'none'}),
                    ],
                    style={"marginTop": "16px"} 
                ),
                html.Label("Select Day of Week", style={"marginTop": "8px"})
            ], 
            width=4
        ),
        dbc.Col(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id='roadway-name-input',
                            options=[{'label': i, 'value': i} for i in df['roadway_name'].unique()],
                            clearable=True
                        ), 
                        html.Div(id='hidden-div-roadway-name', style={'display':'none'}),
                    ],
                    style={"marginTop": "16px"} 
                ),
                html.Label("Select Roadway Name", style={"marginTop": "8px"})
            ], 
            width=4
        )
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='accident-heatmap') 
        ], width=4), 
        dbc.Col([
            dcc.Graph(id='scattermap') 
        ], width=8)  
    ])

    
])
    

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content") 
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    """
    Callback to render the content of the page depending on the current URL.

    Parameters:
    pathname (str): The current URL path.

    Returns:
    html: The HTML layout of the page to be rendered. If the URL path is '/', the home page layout is returned.
          For any other URL path, a 404 error page is returned.
    """
    if pathname == "/":
        return home_page_layout

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognized..."),
        ]
    )

@app.callback(
    Output('hidden-div-day-of-week', 'children'),
    [Input('day-of-week-input', 'value')]
)
def update_hidden_div_day_of_week(entered_day_of_week):
    if entered_day_of_week is None:
        raise PreventUpdate
    return entered_day_of_week  

@app.callback(
    Output('hidden-div-roadway-name', 'children'),
    [Input('roadway-name-input', 'value')]
)
def update_hidden_div_roadway_name(entered_roadway_name):
    if entered_roadway_name is None:
        raise PreventUpdate
    return entered_roadway_name


@app.callback(
    Output('accident-heatmap', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('day-of-week-input', 'value'),
    ]
)
def update_heatmap(selected_start_date, selected_end_date, selected_day_of_week):
    return plot_accident_heatmap(df, selected_start_date, selected_end_date, day=selected_day_of_week)

@app.callback(
    Output('scattermap', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('day-of-week-input', 'value'),
     Input('roadway-name-input', 'value')
    ]
)
def update_scattermap(selected_start_date, selected_end_date, selected_day_of_week, selected_roadway_name):
    return plot_accidents_on_map(df, selected_start_date, selected_end_date, selected_day_of_week, selected_roadway_name)

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




