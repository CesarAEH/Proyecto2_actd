import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import tensorflow as tf
import joblib
import numpy as np

#datos
df = pd.read_csv("Icfes_limpio3.csv")
model = tf.keras.models.load_model("prediccionicfes.h5")
scaler = joblib.load("modelo_icfes1.pkl") 

#app
app = dash.Dash(__name__)
app.title = "Predicción ICFES"

# Layout
app.layout = html.Div([
    html.H1("Tablero de Predicción de Desempeño ICFES", style={'textAlign': 'center'}),

    html.Div([
        html.H3("Ingrese los siguientes datos:"),
        html.Label("Área del colegio(urbano o rural)"),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in df['cole_area_ubicacion'].dropna().unique()],
            id='input_area'
        ),

        html.Label("Carácter del colegio(oficial/no oficial)"),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in df['cole_naturaleza'].dropna().unique()],
            id='input_caracter'
        ),

        html.Label("Departamento"),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in df['cole_depto_ubicacion'].dropna().unique()],
            id='input_departamento'
        ),

        html.Label("¿Es bilingüe el colegio?"),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in df['cole_bilingue'].dropna().unique()],
            id='input_bilingue'
        ),

        html.Label("Estrato de vivienda familia"),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in df['fami_estratovivienda'].dropna().unique()],
            id='input_estrato'
        ),

        html.Br(),
        html.Button("Predecir Nivel", id='predict_button', n_clicks=0),
        html.Div(id='prediction_output', style={'marginTop': 20, 'fontWeight': 'bold'})
    ], style={'width': '30%', 'float': 'left', 'padding': 20}),

    html.Div([
        dcc.Graph(id='bar_area'),
        dcc.Graph(id='map_departamento'),
        dcc.Graph(id='heatmap_estrato_bilingue')
    ], style={'width': '65%', 'float': 'right'})
])

# Visualizaciones
@app.callback(
    Output('bar_area', 'figure'),
    Input('predict_button', 'n_clicks')
)
def update_bar(n):
    fig = px.histogram(df, x='cole_area_ubicacion', color='NIVEL',
                       barmode='group', histfunc='count',
                       title="Distribución de Nivel por Área del Colegio")
    return fig

@app.callback(
    Output('map_departamento', 'figure'),
    Input('predict_button', 'n_clicks')
)
def update_map(n):
    dpt = df.groupby("cole_depto_ubicacion")["punt_global"].mean().reset_index()
    fig = px.bar(dpt, x="cole_depto_ubicacion", y="punt_global",
                 title="Promedio Puntaje Global por Departamento")
    return fig

@app.callback(
    Output('heatmap_estrato_bilingue', 'figure'),
    Input('predict_button', 'n_clicks')
)
def update_heatmap(n):
    cross = pd.crosstab(df['fami_estratovivienda'], df['cole_bilingue'], normalize='index')
    fig = px.imshow(cross, text_auto=True, title="Proporción de Colegios Bilingües por Estrato")
    return fig

# Predicción
@app.callback(
    Output('prediction_output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('input_area', 'value'),
    State('input_caracter', 'value'),
    State('input_departamento', 'value'),
    State('input_bilingue', 'value'),
    State('input_estrato', 'value')
)
def predict_level(n_clicks, area, caracter, dep, bilingue, estrato):
    if None in [area, caracter, dep, bilingue, estrato]:
        return "Llene todos los campos."

    # Limpiar espacios y construir input
    input_dict = {
        'cole_area_ubicacion': area.strip(),
        'cole_naturaleza': caracter.strip(),
        'cole_depto_ubicacion': dep.strip(),
        'cole_bilingue': bilingue.strip(),
        'fami_estratovivienda': estrato.strip()
    }

    input_df = pd.DataFrame([input_dict])

    # Convertir a dummies con mismo orden que entrenamiento
    dummy_df = pd.get_dummies(pd.concat([df[input_dict.keys()], input_df], axis=0), drop_first=True)
    input_transformed = dummy_df.tail(1)

    # Escalar
    input_scaled = scaler.transform(input_transformed)

    # Predecir
    prediction = model.predict(input_scaled)
    level = np.argmax(prediction, axis=1)[0] + 1
    nivel_map = {1: 'Bajo', 2: 'Medio', 3: 'Alto'}

    return f"Predicción: El estudiante tiene un desempeño {nivel_map[level]}"

# Ejecutar localmente
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
