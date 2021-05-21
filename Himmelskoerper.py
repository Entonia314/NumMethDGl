from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import SupplementaryFiles.dash_reusable_components as drc


def mult(vector, scalar):
    newvector = [0] * len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i] * scalar
    return newvector


[x0, y0], [v0, w0] = [-0.0347, 1.1856], [0.2495, -0.1076]
[x1, y1], [v1, w1] = [0.2693, -1.0020], [0.2059, -0.9396]
[x2, y2], [v2, w2] = [-0.2328, -0.5978], [-0.4553, 1.0471]
inits1 = np.array([[[x0, y0], [v0, w0]], [[x1, y1], [v1, w1]], [[x2, y2], [v2, w2]]])
inits2 = np.array([[[-0.602885898116520, 1.059162128863347 - 1], [0.122913546623784, 0.747443868604908]],
                   [[0.252709795391000, 1.058254872224370 - 1], [-0.019325586404545, 1.369241993562101]],
                   [[-0.355389016941814, 1.038323764315145 - 1], [-0.103587960218793, -2.116685862168820]]])
inits3 = np.array([[[0.716248295712871, 0.384288553041130], [1.245268230895990, 2.444311951776573]],
                   [[0.086172594591232, 1.342795868576616], [-0.675224323690062, -0.962879613630031]],
                   [[0.538777980807643, 0.481049882655556], [-0.570043907205925, -1.481432338146543]]])
r = 0.5
inits4 = np.array([[[1, 0], mult([0, 1], r)],
                   [[-0.5, 3 ** (1 / 2) / 2], mult([-3 ** (1 / 2) / 2, -0.5], r)],
                   [[-0.5, -3 ** (1 / 2) / 2], mult([3 ** (1 / 2) / 2, -0.5], r)]])
p51 = 0.347111
p52 = 0.532728
inits5 = np.array([[[-1, 0], [p51, p52]],
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

mass = [1, 1, 0.0000000000001]
mass_szenario4 = [0, 0, 0.00000001]
G = 1
t_start = 0
t_stop = 10

method_dict = {'forward_euler': 'Explizites Euler-Verfahren', 'backward_euler': 'Implizites Euler-Verfahren'}
init_dict = {1: inits1, 2: inits2, 3: inits3, 4: inits4, 5: inits5, 6: inits6, 7: inits7}

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
                'Das Drei-Körper-Problem',
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
                                id='second-card',
                                style={'margin': '10px 10px 10px 10px'},
                                children=[
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
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=1,
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
                                    drc.NamedDropdown(
                                        name='Masse von Objekt 1',
                                        id='mass1-dropdown',
                                        options=[
                                            {
                                                'label': 'Sonne',
                                                'value': 1.9884 * 10 ** 30
                                            },
                                            {
                                                'label': 'Saturn',
                                                'value': 5.683 * 10 ** 26
                                            },
                                            {
                                                'label': 'Erde',
                                                'value': 5.9724 * 10 ** 24
                                            },
                                            {
                                                'label': 'Mond',
                                                'value': 1.9884 * 10 ** 30
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=1.9884 * 10 ** 30,
                                    ),
                                    drc.NamedDropdown(
                                        name='Masse von Objekt 2',
                                        id='mass2-dropdown',
                                        options=[
                                            {
                                                'label': 'Sonne',
                                                'value': 1.9884 * 10 ** 30
                                            },
                                            {
                                                'label': 'Saturn',
                                                'value': 5.683 * 10 ** 26
                                            },
                                            {
                                                'label': 'Erde',
                                                'value': 5.9724 * 10 ** 24
                                            },
                                            {
                                                'label': 'Mond',
                                                'value': 7.346 * 10 ** 22
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=5.9724 * 10 ** 24,
                                    ),
                                    drc.NamedDropdown(
                                        name='Masse von Objekt 3',
                                        id='mass3-dropdown',
                                        options=[
                                            {
                                                'label': 'Sonne',
                                                'value': 1.9884 * 10 ** 30
                                            },
                                            {
                                                'label': 'Saturn',
                                                'value': 5.683 * 10 ** 26
                                            },
                                            {
                                                'label': 'Erde',
                                                'value': 5.9724 * 10 ** 24
                                            },
                                            {
                                                'label': 'Mond',
                                                'value': 7.346 * 10 ** 22
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=7.346 * 10 ** 22,
                                    ),
                                ],
                            ),
                            dbc.Card(
                                id='neat-info-card',
                                style={'margin': '10px 10px 10px 10px'},
                                children=[
                                    dbc.CardBody([
                                        html.H4('Modellannahmen'),
                                        html.Li('Masse von Objekt 1 = 1'),
                                        html.Li('Masse von Objekt 2 = 1'),
                                        html.Li('Masse von Objekt 3 = 1'),
                                        html.Li('Gravitationskonstante g = 1'),
                                    ], style={'margin': '0px 10px 10px 10px'})
                                ]
                            )
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
    Input(component_id='scenario-dropdown', component_property='value'),
    Input(component_id='h-dropdown', component_property='value'),
    Input(component_id='time', component_property='value')
)
def update_masses(number, h, t_end):
    init_data = init_dict[number]
    fig = generate_figures(forward_euler(f, init_data, t_start, t_end, h))
    return fig


# Parameters
AU = 1.496e+11
y_init = [np.array([1, 5]), np.array([-4, 2]), np.array([14, -3])]

h1 = 0.1
h2 = 0.01
N = int((t_stop - t_start) / h1)
mass1 = [1.989e+30, 5.972e+24, 1.898e+27]


def f(t, y):
    g = G
    m = mass
    d0 = ((-g * m[0] * m[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * m[0] * m[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / m[0]
    d1 = ((-g * m[1] * m[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * m[1] * m[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / m[1]
    d2 = ((-g * m[2] * m[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * m[2] * m[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / m[2]
    return np.array([d0, d1, d2])


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
        if g(x0) == 0.0:
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = abs(f(x1)) > e
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
    g = G
    m = mass
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
            t = t + h

            def equation(x):
                e0 = y[i, k - 1, 0] + h * (((-g * m[0] * m[1] * (y[i, k-1, 0] - y[i, k-1, 1]) / np.linalg.norm(y[i, k-1, 0] - y[i, k-1, 1]) ** 3) +
                      (-g * m[0] * m[2] * (y[i, k-1, 0] - y[i, k-1, 2]) / np.linalg.norm(y[i, k-1, 0] - y[i, k-1, 2]) ** 3)) / m[0]) - x
                e1 = y[i, k - 1, 0] + h * (((-g * m[1] * m[2] * (y[i, k-1, 1] - y[i, k-1, 2]) / np.linalg.norm(y[i, k-1, 1] - y[i, k-1, 2]) ** 3) + (
                        -g * m[1] * m[0] * (y[i, k-1, 1] - y[i, k-1, 0]) / np.linalg.norm(y[i, k-1, 1] - y[i, k-1, 0]) ** 3)) / m[1]) - x
                e2 = y[i, k - 1, 0] + h * (((-g * m[2] * m[0] * (y[i, k-1, 2] - y[i, k-1, 0]) / np.linalg.norm(y[i, k-1, 2] - y[i, k-1, 0]) ** 3) + (
                        -g * m[2] * m[1] * (y[i, k-1, 2] - y[i, k-1, 1]) / np.linalg.norm(y[i, k-1, 2] - y[i, k-1, 1]) ** 3)) / m[2]) - x
                return [e0, e1, e2]

            def equation_diff(x):
                return h * (-4) * t * x - 1

            y[i, k, 0] = newton_raphson(equation, equation_diff, y[i, k - 1, 0], 0.0001, 10)
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
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        y[:, k + 1, :] = fpi(f, t, y[:, k, :], h, 5)
        t = t + h
        t_list[k + 1] = t
    return y, t_list


def generate_figures(method):
    y, t = method
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name='Objekt 1',
                         line=dict(width=2, color="blue")),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name='Objekt 2',
                         line=dict(width=2, color="red")),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name='Objekt 3',
                         line=dict(width=2, color="green"))],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title='Explizites Euler-Verfahren', hovermode="closest"),
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
