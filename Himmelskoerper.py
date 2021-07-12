from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from scipy.optimize import fsolve
import SupplementaryFiles.dash_reusable_components as drc

t_start = 0
mass = [1, 1, 1]
g = 1


# Auxiliary function to multiply list with scalar
def mult(vector, scalar):
    newvector = [0] * len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i] * scalar
    return newvector


# Initial conditions for simplified models

[x0, y0], [v0, w0] = [-0.0347, 1.1856], [0.2495, -0.1076]
[x1, y1], [v1, w1] = [0.2693, -1.0020], [0.2059, -0.9396]
[x2, y2], [v2, w2] = [-0.2328, -0.5978], [-0.4553, 1.0471]
inits4 = np.array([[[x0, y0], [v0, w0]], [[x1, y1], [v1, w1]], [[x2, y2], [v2, w2]]])

inits1 = np.array([[[-0.602885898116520, 1.059162128863347 - 1], [0.122913546623784, 0.747443868604908]],
                   [[0.252709795391000, 1.058254872224370 - 1], [-0.019325586404545, 1.369241993562101]],
                   [[-0.355389016941814, 1.038323764315145 - 1], [-0.103587960218793, -2.116685862168820]]])

inits5 = np.array([[[0.716248295712871, 0.384288553041130], [1.245268230895990, 2.444311951776573]],
                   [[0.086172594591232, 1.342795868576616], [-0.675224323690062, -0.962879613630031]],
                   [[0.538777980807643, 0.481049882655556], [-0.570043907205925, -1.481432338146543]]])

r = 0.5
inits2 = np.array([[[1, 0], mult([0, 1], r)],
                   [[-0.5, 3 ** (1 / 2) / 2], mult([-3 ** (1 / 2) / 2, -0.5], r)],
                   [[-0.5, -3 ** (1 / 2) / 2], mult([3 ** (1 / 2) / 2, -0.5], r)]])

p51 = 0.347111
p52 = 0.532728
inits3 = np.array([[[-1, 0], [p51, p52]],
                   [[1, 0], [p51, p52]],
                   [[0, 0], [-2 * p51, -2 * p52]]])

p61 = 0.464445
p62 = 0.396060
inits6 = np.array([[[-1, 0], [p61, p62]],
                   [[1, 0], [p61, p62]],
                   [[0, 0], [-2 * p61, -2 * p62]]])

p71 = 0.080584
p72 = 0.588836
inits7 = np.array([[[-1, 0], [p71, p72]],
                   [[1, 0], [p71, p72]],
                   [[0, 0], [-2 * p71, -2 * p72]]])


# Initial conditions for real life models

Mear = 6e24  # Mass of Earth in kg
Msun = 2e30  # Mass of Sun in kg
Mjup = 1.9e27  # Mass of Jupiter
Msat = 5.7e26  # Mass of Saturn

G = 6.673e-11  # Gravitational Constant

RR = 1.496e11  # Normalizing distance in km (= 1 AU)
MM = 6e24  # Normalizing mass
TT = 365 * 24 * 60 * 60.0  # Normalizing time (1 year)

FF = (G * MM ** 2) / RR ** 2  # Unit force
EE = FF * RR  # Unit energy

GG = (MM * G * TT ** 2) / (RR ** 3)

Mear = Mear / MM  # Normalized mass of Earth
Msun = Msun / MM  # Normalized mass of Sun
Mjup = 500 * Mjup / MM  # Normalized mass of Jupiter/Super Jupiter
Msat = Msat / MM  # Normalized mass of Saturn

ti = 0  # initial time = 0
tf = 120  # final time = 120 years

N = 100 * tf  # 100 points per year
t = np.linspace(ti, tf, N)  # time array from ti to tf with N points

h = t[2] - t[1]  # time step (uniform)

# Initialization

rear = [1496e8 / RR, 0]  # initial position of earth
rjup = [5.2, 0]  # initial position of Jupiter
rsat = [9.582, 0]
rsun = [0, 0]

magear = np.sqrt(Msun * GG / rear[0])  # Magnitude of Earth's initial velocity
magjup = 13.06e3 * TT / RR  # Magnitude of Jupiter's initial velocity
magsat = 9.68e3 * TT / RR  # Magnitude of Saturn's initial velocity

vear = [0, magear * 1.0]  # Initial velocity vector for Earth.Taken to be along y direction as ri is on x axis.
vjup = [0, magjup * 1.0]  # Initial velocity vector for Jupiter
vsat = [0, magsat * 1.0]  # Initial velocity vector for Saturn
vsun = [0, 0]

inits8 = np.array([[rsun, vsun],
                   [rsat, vsat],
                   [rear, vear]])

mass_szenario4 = [0, 0, 0.00000001]

method_dict = {'forward_euler': 'Explizites Euler-Verfahren', 'backward_euler': 'Implizites Euler-Verfahren'}
init_dict = {1: inits1, 2: inits2, 3: inits3, 4: inits4, 5: inits5, 6: inits6, 7: inits7, 8: inits8}
mass_dict = {'sun': Msun, 'sat': Msat, 'jup': Mjup, 'ear': Mear}
r_dict = {'sun': rsun, 'sat': rsat, 'jup': rjup, 'ear': rear}
v_dict = {'sun': vsun, 'sat': vsat, 'jup': vjup, 'ear': vear}
colour_dict = {'sun': 'yellow', 'sat': 'grey', 'jup': 'orange', 'ear': 'blue'}
name_dict = {'sun': 'Sonne', 'sat': 'Saturn', 'jup': 'Jupiter', 'ear': 'Erde'}


# Here is were the dash app begins
# Standard css style sheet recommended by Dash
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Generates the Dash app
server = Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets
                )
app.title = 'Himmelsmechanik'

app.layout = dbc.Container(fluid=True, style={'background-color': '#333399'}, children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", style={'background-color': '#333399'}, children=[
        html.Div(className='container scalable', children=[
            html.H2(html.A(
                'Himmelsmechanik',
                href='',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                    'background-color': '#333399',
                    'text-align': 'center'
                }
            )),
        ]),
    ]),

    html.Div(
        [
            dbc.Row(style={'background-color': 'white', 'margin': '0px 0px 0px 0px'},
                    children=[
                        dbc.Col([
                            dbc.Card(
                                id='first-card',
                                style={'margin': '10px 10px 10px 10px'},
                                children=[
                                    html.H3('Parameter', style={'margin': '10px 10px 0px 0px'}, ),
                                    drc.NamedRadioItems(
                                        name='Modell',
                                        id="radios",
                                        options=[
                                            {"label": "Simplifiziertes Modell", "value": 1},
                                            {"label": "Sonnensystem", "value": 2},
                                        ],
                                        value=1,
                                        style={'margin': '10px 10px 10px 10px'},
                                    ),
                                    drc.NamedDropdown(
                                        name='Schrittweite',
                                        id='h-dropdown',
                                        options=[
                                            {
                                                'label': 'h = 0.1',
                                                'value': 0.1
                                            },
                                            {
                                                'label': 'h = 0.01',
                                                'value': 0.01
                                            },
                                            {
                                                'label': 'h = 0.001',
                                                'value': 0.001
                                            },
                                            {
                                                'label': 'h = 0.0005',
                                                'value': 0.0005
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=0.01,
                                    ),
                                    drc.NamedSlider(
                                        name='Zeit',
                                        id='time',
                                        min=0,
                                        max=30,
                                        step=1,
                                        marks={0: '0', 5: '5', 10: '10',
                                               15: '15', 20: '20', 25: '25', 30: '30'},
                                        value=10
                                    ),
                                    drc.NamedRadioItems(
                                        name='Anzeigen',
                                        id="error",
                                        options=[
                                            {"label": "Bahnen", "value": 1},
                                            {"label": "Fehler in Vergleich zu Runge-Kutta", "value": 2},
                                        ],
                                        value=1,
                                        style={'margin': '10px 10px 10px 10px'},
                                    ),
                                ]),
                            dbc.Tabs([
                                dbc.Tab(label="Simplifiziertes Modell", children=[
                                    dbc.Card(
                                        id='second-card',
                                        style={'margin': '10px 10px 10px 10px'},
                                        children=[
                                            html.H5('Simplifiziertes Modell', style={'margin': '10px 10px 10px 10px'}),
                                            html.Li('Modellannahme: g = 1', style={'margin': '5px 5px 5px 25px'}),
                                            drc.NamedDropdown(
                                                name='Szenario',
                                                id='scenario-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Szenario 1',
                                                        'value': 1
                                                    },
                                                    {
                                                        'label': 'Szenario 2',
                                                        'value': 2
                                                    },
                                                    {
                                                        'label': 'Szenario 3',
                                                        'value': 3
                                                    },
                                                    {
                                                        'label': 'Szenario 4',
                                                        'value': 4
                                                    },
                                                    {
                                                        'label': 'Szenario 5',
                                                        'value': 5
                                                    },
                                                    {
                                                        'label': 'Szenario 6',
                                                        'value': 6
                                                    },
                                                    {
                                                        'label': 'Szenario 7',
                                                        'value': 7
                                                    },
                                                    {
                                                        'label': 'Szenario 8',
                                                        'value': 8
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value=1,
                                            ),
                                            drc.NamedSlider(
                                                name='Masse von Objekt 1',
                                                id='simple_mass1',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                            drc.NamedSlider(
                                                name='Masse von Objekt 2',
                                                id='simple_mass2',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                            drc.NamedSlider(
                                                name='Masse von Objekt 3',
                                                id='simple_mass3',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                        ])],
                                        ),
                                dbc.Tab(label="Sonnensystem", children=[
                                    dbc.Card(
                                        id='third-card',
                                        style={'margin': '10px 10px 10px 10px'},
                                        children=[
                                            html.H5('Sonnensystem', style={'margin': '10px 10px 10px 10px'}),
                                            drc.NamedDropdown(
                                                name='Objekt 1',
                                                id='object1-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sonne',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Erde',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'Mond',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='sun',
                                            ),
                                            drc.NamedDropdown(
                                                name='Objekt 2',
                                                id='object2-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sonne',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Erde',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'Mond',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='ear',
                                            ),
                                            drc.NamedDropdown(
                                                name='Objekt 3',
                                                id='object3-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sonne',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Erde',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'Mond',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='jup',
                                            ),
                                        ],
                                    ),
                                ])]),
                        ],
                            width=3),
                        dbc.Col(children=[
                            html.Div(
                                id='div-expl-euler',
                                children=dcc.Graph(
                                    id='expl-euler'
                                ),
                            ),
                            html.Div(
                                id='div-runge-kutta',
                                children=dcc.Graph(
                                    id='runge-kutta'
                                )
                            ),
                        ], width=4, align='center'),
                        dbc.Col(children=[
                            html.Div(
                                id='div-impl-euler',
                                children=dcc.Graph(
                                    id='impl-euler'
                                ),
                            ),
                            html.Div(
                                id='div-corr-pred',
                                children=dcc.Graph(
                                    id='corr-pred'
                                )
                            ),
                        ], width=4, align='center'),
                    ]),
        ]
    ),
]
                           )


@app.callback(
    Output(component_id='expl-euler', component_property='figure'),
    Output(component_id='impl-euler', component_property='figure'),
    Output(component_id='runge-kutta', component_property='figure'),
    Output(component_id='corr-pred', component_property='figure'),
    Input(component_id='radios', component_property='value'),
    Input(component_id='h-dropdown', component_property='value'),
    Input(component_id='time', component_property='value'),
    Input(component_id='scenario-dropdown', component_property='value'),
    Input(component_id='simple_mass1', component_property='value'),
    Input(component_id='simple_mass2', component_property='value'),
    Input(component_id='simple_mass3', component_property='value'),
    Input(component_id='object1-dropdown', component_property='value'),
    Input(component_id='object2-dropdown', component_property='value'),
    Input(component_id='object3-dropdown', component_property='value')
)
def update_figure(model, h, t_end, scenario, m1, m2, m3, o1, o2, o3):
    global mass
    global g
    if model == 1:
        names = ['Objekt 1', 'Objekt 2', 'Objekt 3']
        colours = ['red', 'blue', 'violet']
        init_data = init_dict[scenario]
        mass = [m1, m2, m3]
        g = 1
        fig_expl = generate_figures(forward_euler(f, init_data, t_start, t_end, h), 'Explizites Euler-Verfahren', names,
                                    colours)
        fig_rk = generate_figures(runge_kutta(f, init_data, t_start, t_end, h), 'Runge-Kutta-Verfahren der Stufe 3',
                                  names, colours)
        try:
            fig_impl = generate_figures(backward_euler1(f, init_data, t_start, t_end, h), 'Implizites Euler-Verfahren',
                                        names, colours)
        except:
            fig_impl = fig_not_convergent('Implizites Euler-Verfahren')
        try:
            fig_precor = generate_figures(predictor_corrector(f, init_data, t_start, t_end, h),
                                          'Prädiktor-Korrektor-Verfahren',
                                          names, colours)
        except:
            fig_precor = fig_not_convergent('Prädiktor-Korrektor-Verfahren')
        return fig_expl, fig_impl, fig_rk, fig_precor
    elif model == 2:
        g = GG
        names = [name_dict[o1], name_dict[o2], name_dict[o3]]
        colours = [colour_dict[o1], colour_dict[o2], colour_dict[o3]]
        mass = np.array([mass_dict[o1], mass_dict[o2], mass_dict[o3]])
        init_data = np.array([[r_dict[o1], v_dict[o1]], [r_dict[o2], v_dict[o2]], [r_dict[o3], v_dict[o3]]])
        fig_expl = generate_figures(forward_euler(f, init_data, t_start, t_end, h),
                                    'Explizites Euler-Verfahren', names, colours)
        fig_rk = generate_figures(runge_kutta(f, init_data, t_start, t_end, h),
                                  'Runge-Kutta-Verfahren der Stufe 3', names, colours)
        try:
            fig_impl = generate_figures(backward_euler1(f, init_data, t_start, t_end, h), 'Implizites Euler-Verfahren',
                                        names, colours)
        except:
            fig_impl = fig_not_convergent('Implizites Euler-Verfahren')
        try:
            fig_precor = generate_figures(predictor_corrector(f, init_data, t_start, t_end, h),
                                          'Prädiktor-Korrektor-Verfahren',
                                          names, colours)
        except:
            fig_precor = fig_not_convergent('Prädiktor-Korrektor-Verfahren')
        return fig_expl, fig_impl, fig_rk, fig_precor


def f(t, y):
    d0 = ((-g * mass[0] * mass[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / mass[0]
    d1 = ((-g * mass[1] * mass[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * mass[1] * mass[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / mass[1]
    d2 = ((-g * mass[2] * mass[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * mass[2] * mass[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / mass[2]
    return np.array([d0, d1, d2])


def flong(y):
    d00 = ((-g * mass[0] * mass[1] * (y[0] - y[2]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) +
           (-g * mass[0] * mass[2] * (y[0] - y[4]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d01 = ((-g * mass[0] * mass[1] * (y[1] - y[3]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) + (
            -g * mass[0] * mass[2] * (y[1] - y[5]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d10 = ((-g * mass[1] * mass[2] * (y[2] - y[4]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[2] - y[0]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d11 = ((-g * mass[1] * mass[2] * (y[3] - y[5]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[3] - y[1]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d20 = ((-g * mass[2] * mass[0] * (y[4] - y[0]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[4] - y[2]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    d21 = ((-g * mass[2] * mass[0] * (y[5] - y[1]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[5] - y[3]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    return np.array([d00, d01, d10, d11, d20, d21])


def forward_euler(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            y[i, k + 1, :] = y[i, k, :] + h * v[i, k, :]
            v[i, k + 1, :] = v[i, k, :] + h * f(t, y[:, k, :])[i]
        t = t + h
        t_list[k + 1] = t
    return y, t


def runge_kutta(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            k1 = f(t, y[:, k, :])[i]
            k2 = f(t + 0.5 * h, (y[:, k, :] + h * k1 / 2))[i]
            k3 = f(t + h, (y[:, k, :] + h * (-k1 + 2 * k2)))[i]
            v[i, k + 1] = v[i, k, :] + h / 6 * (k1 + 4 * k2 + k3)
            k12 = v[i, k, :]
            k22 = v[i, k, :] + h * k1 / 2
            k32 = v[i, k, :] + h * (-k1 + 2 * k2)
            y[i, k + 1] = y[i, k, :] + h / 6 * (k12 + 4 * k22 + k32)
        t = t + h
        t_list[k + 1] = t
    return y, t_list


def backward_euler_equation(y, yn, h, f):
    ret = y - yn - h * f(y)
    return ret


def backward_euler_scipy(f, y0, t0, t1, h):
    N = int(np.floor((t1 - t0) / h))
    t = t0
    v = np.zeros((N + 1, 6))
    y = np.zeros((N + 1, 6))
    y_start = y0[:, 0, :]
    v_start = y0[:, 1, :]
    y0_reshape = y_start.reshape(6, )
    v0_reshape = v_start.reshape(6, )
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[0, :] = v0_reshape
    y[0, :] = y0_reshape

    for k in range(N):
        v[k + 1, :] = fsolve(backward_euler_equation, y[k, :], (y[k, :], h, flong))
        y[k + 1, :] = y[k, :] + h * v[k, :]
        t = t + h
        t_list[k + 1] = t

    y_ret = np.zeros((len(y0), N + 1, 2))
    y_ret[:, :, 0] = y[:, ::2].transpose()
    y_ret[:, :, 1] = y[:, 1::2].transpose()
    return y_ret, t_list


def newton_raphson(f, g, x0, e, N):
    """
    Numerical solver of the equation f(x) = 0
    :param f: Function, left side of equation f(x) = 0 to solve
    :param g: Function, derivative of f
    :param x0: Float, initial guess
    :param e: Float, tolerable error
    :param N: Integer, maximal steps
    :return:
    """
    step = 1
    flag = 1
    condition = True
    while condition:
        if np.all(g(x0) == 0.0):
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = np.any(abs(f(x1)) > e)
    if flag == 1:
        return x1
    else:
        print('\nNot Convergent.')


def backward_euler1(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(1, N + 1):

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)

                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / np.linalg.norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            v[i, k, :] = newton_raphson(fixpoint, fixpoint_deriv, y[:, k - 1, :], 0.0001, 30)[i, :]
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :]
        t = t + h
        t_list[k] = t
    return y, t_list


def predictor_corrector(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(1, N + 1):
        vp = v[:, k - 1, :] + h * f(t, y[:, k - 1, :])
        yp = y[:, k - 1, :] + h * v[:, k - 1, :]

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)
                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / np.linalg.norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            vc = newton_raphson(fixpoint, fixpoint_deriv, yp, 0.0001, 10)[i, :]
            v[i, k, :] = vc
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :]
        t = t + h
        t_list[k] = t
    return y, t_list


def fpi(f, t, yk, h, steps):
    x = yk
    while steps > 0:
        x = (1 + h) * x + h ** 2 * f(t, x)
        steps += -1
    return x


def backward_euler(f, y0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        v[:, k + 1, :] = fpi(f, t, y[:, k, :], h, 5)
        y[:, k + 1, :] = y[:, k, :] + h * v[:, k, :]
        t = t + h
        t_list[k + 1] = t
    return y, t_list


def generate_figures(method, title, names, colours):
    y, t = method
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name=names[0],
                         line=dict(width=2, color=colours[0])),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name=names[1],
                         line=dict(width=2, color=colours[1])),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name=names[2],
                         line=dict(width=2, color=colours[2]))],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title=title, hovermode="closest"),
    )
    return fig


def generate_error_figures(method, title, names, colours):
    y_rk, t_rk = runge_kutta
    y_meth, t_meth = method
    y = np.absolute(y_rk - y_meth)
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name=names[0],
                         line=dict(width=2, color=colours[0])),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name=names[1],
                         line=dict(width=2, color=colours[1])),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name=names[2],
                         line=dict(width=2, color=colours[2]))],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title=title, hovermode="closest"),
    )
    return fig


def generate_animations(method, title, names, colours):
    y, t = method
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         name="frame",
                         mode="lines",
                         line=dict(width=2, color="blue")),
              go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         name="curve",
                         mode="lines",
                         line=dict(width=2, color="blue"))
              ],
        layout=go.Layout(width=600, height=600,
                         xaxis=dict(autorange=True, zeroline=False),
                         yaxis=dict(autorange=True, zeroline=False),
                         title="Moving Frenet Frame Along a Planar Curve",
                         hovermode="closest",
                         updatemenus=[dict(type="buttons",
                                           buttons=[dict(label="Play",
                                                         method="animate",
                                                         args=[None])])]),

        frames=[go.Frame(
            data=[go.Scatter(
                x=[xx[k], xend[k], None, xx[k], xnoe[k]],
                y=[yy[k], yend[k], None, yy[k], ynoe[k]],
                mode="lines",
                line=dict(color="red", width=2))
            ]) for k in range(N)]
    )
    return fig


def fig_not_convergent(title):
    fig = px.scatter(x=[1, 1, 1, 1, 1, 1, 1, 1], y=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 0.75],
                     title=str(title + ': Nicht konvergent'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
