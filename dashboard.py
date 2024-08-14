#--------------------------------------------------------------------------------------------------------------
# Dashboard de Simulacion, Valoracion de Opciones y Riesgos
# Autor: Pablo Marchesi Selma
# Universidad Politecnica de Valencia 
# Junio 2024
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
# LIBRERIAS
#--------------------------------------------------------------------------------------------------------------

import dash
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import math 
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------------------------------
# FUNCIONES 
#--------------------------------------------------------------------------------------------------------------

# Obten la fecha de hace n dias
def date_n_days_ago(n):
    today = datetime.now()
    delta = timedelta(days=n)
    result = today - delta
    formatted_result = result.strftime('%Y-%m-%d')

    return formatted_result

# Extrae datos de yfinance
def get_stock_data(stock_symbol, n):
    stock = yf.download(stock_symbol, start = date_n_days_ago(n), progress = False)

    return stock


def JumpDiff_Merton(S0, mu, sigma, lamb, mu_J, sigma_J, steps, paths, Delta_t):
    
    # Correccion del drift para mantener neutralidad al riesgo
    r_J = lamb*(np.exp(mu_J + 0.5*sigma_J**2)-1)     

    Z1 = np.random.standard_normal((steps+1, paths))
    Z2 = np.random.standard_normal((steps+1, paths))
    Y = np.random.poisson(lamb*Delta_t, (steps+1, paths))

    cum_poi = np.multiply(Y, mu_J + sigma_J*Z2).cumsum(axis = 0)
    gbm = np.cumsum(((mu -  sigma**2/2 -r_J)*Delta_t+ sigma*np.sqrt(Delta_t) * Z1), axis=0)
    
    return np.exp(cum_poi + gbm)*S0

# PDF del proceso salto-difusion vectorizada
def jump_diffusion_pdf_vector(x, Delta_t, mu, sigma, lambd, mu_J, sigma_J):
    k = np.arange(100) 
    t = np.array([math.factorial(f) for f in k])
    pk = np.exp(-lambd * Delta_t) * ((lambd * Delta_t) ** k) / t
    mu_phi = (mu - (sigma ** 2) / 2) * Delta_t + mu_J * k
    sigma_phi = np.sqrt((sigma ** 2) * Delta_t + (sigma_J ** 2) * k)
    
    pdf_contributions = pk * norm.pdf(x, loc=mu_phi, scale=sigma_phi)
    pdf = sum(pdf_contributions)  
    
    return np.log(pdf)

# Funcion de verosimilitud
def log_likelihood(theta, rets, Delta_t):
    mu, sigma, lambd, mu_J, sigma_J = theta
    lnL = 0
    
    for x in rets:
        pdf = jump_diffusion_pdf_vector(x, Delta_t, mu, sigma, lambd, mu_J, sigma_J)
        lnL += pdf
        
    return -lnL

# Calcula el VaR para un intervalo de confianza dado
def calculate_var(returns, alpha):
    var = np.percentile(returns, alpha * 100)

    return var

# Funcion para calcular el precio de la opcion vía montecarlo (T en días)
def mc_option_valuation(S0, strikes, T, r, mu, sigma, lambd, mu_J, sigma_J, paths, option_type='call'):
    S_t = JumpDiff_Merton(S0, mu, sigma, lambd, mu_J, sigma_J, steps = T, paths = paths, Delta_t = 1/252)
    S_T = S_t[-1]
    
    results = {}
    for K in strikes:
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'")
        
        disc_payoffs = np.exp(-r * T/365) * payoffs
        option_price = np.mean(disc_payoffs)
        results[K] = option_price
    
    return results

#---------------------------------------------------------------------------------------------------------
# CONFIGURACIÓN DEL DASHBOARD
#---------------------------------------------------------------------------------------------------------

# Inicializacion de la app de Dash
app = dash.Dash(__name__)
app.title = 'Dashboard'

# Apariencia del dashboard
app.layout = html.Div([
    html.H1("Dashboard de Simulación, Riesgos y Valoración de Opciones"),

    html.Div([
        html.Label('Ticker: '),
        dcc.Input(id='input-stock', type='text', value='MSFT', style={'width': '45px', 'margin-right': '10px'}),
        
        html.Label('Días: '),
        dcc.Input(id='input-steps', type='number', value=500, style={'width': '45px', 'margin-right': '10px'}),
        
        html.Label('Trayectorias: '),
        dcc.Input(id='input-paths', type='number', value = 5000, style={'width': '55px', 'margin-right': '10px'}),

        html.Label('Opción: '),
        dcc.Input(id='input-option_type', type='text', value='call', style={'width': '40px', 'margin-right': '10px'}),

        html.Label('Tipo de Interés: '),
        dcc.Input(id='input-r', type='number', value = 0.055, step = 0.001, style={'width': '50px', 'margin-right': '10px'}),

        html.Label('Nivel de Confianza: '),
        dcc.Input(id='input-alpha', type='number', value = 0.95, step = 0.01, style={'width': '50px', 'margin-right': '10px'}),

        html.Button('Simular', id ='simulate-button', n_clicks=0),

        ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='stock-graph', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='sim-graph', style={'width': '50%', 'display': 'inline-block'}),
    ]),

    html.Div([
    dcc.Graph(id='var-histogram', style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='option-valuation', style={'width': '50%', 'display': 'inline-block'}),
    ]),
])

# Callback para actualizar el dashboard segun los inputs
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('sim-graph', 'figure'),
     Output('option-valuation','figure'),
     Output('var-histogram', 'figure')],
    [Input('simulate-button', 'n_clicks')],
    [State('input-stock', 'value'),
     State('input-steps', 'value'),
     State('input-paths', 'value'),
     State('input-option_type', 'value'),
     State('input-r', 'value'),
     State('input-alpha', 'value')])

#---------------------------------------------------------------------------------------------------------
# ACTUALIZACION DEL DASHBOARD
#---------------------------------------------------------------------------------------------------------

def update_graph(n_clicks, stock_symbol, steps, paths, option_type, r, alpha):

    if n_clicks >= 0:

        # Definimos algunos parametros iniciales
        Delta_t = 1/252
        alpha = 1 - alpha 

        # Descarga de datos 
        S = get_stock_data(stock_symbol, steps)
        S['Log Rets'] = np.log(S['Close']/S['Close'].shift(1))
        S0 = S['Close'].iloc[-1]

        # Separamos en retornos con salto y sin salto
        eps = 3*np.std(S['Log Rets'])
        R_J = S[np.abs(S['Log Rets']) >= eps]['Log Rets']
        R_D = S[np.abs(S['Log Rets']) < eps]['Log Rets']

        # Estimamos los parametros mu y sigma a partir de los retornos sin salto
        u_D = np.mean(R_D); s_D = np.var(R_D, ddof = 1)
        mu_0 = (u_D + 0.5*s_D)/Delta_t
        sigma_0 = np.sqrt(s_D/Delta_t)

        # Estimamos lambda (en saltos/año)
        lambd_0 = len(R_J)/(len(S)*Delta_t)

        # Estimamos mu_J y sigma_J a partir de los retornos con salto
        u_J = np.mean(R_J); s_J = np.var(R_J, ddof = 1)
        mu_J_0 = u_J - u_D
        sigma_J_0 = np.sqrt(s_J - s_D)

        # Parametros inciales y restricciones
        theta_0 = [mu_0, sigma_0, lambd_0, mu_J_0, sigma_J_0]
        bounds = [(-1.5, 1.5), (0.01, 1), (0.5, 50), (-0.5, 0.5), (0.01, 0.5)]

        # Optimizacion
        res = minimize(log_likelihood, theta_0, args=(S['Log Rets'], Delta_t), method='L-BFGS-B', bounds=bounds,options={"maxiter":20})
        mu, sigma, lambd, mu_J, sigma_J = res.x 

        # Grafico de la accion
        stock_fig = go.Figure()
        stock_fig.add_trace(go.Scatter(x = S.index, y = S['Close'], mode='lines', name='Stock Price'))
        stock_fig.update_layout(title=f"Cotización de {stock_symbol}", xaxis_title="Fecha", yaxis_title="Precio Accion ($)")
        
        # Grafico simulaciones 
        S_t = JumpDiff_Merton(S0, mu, sigma, lambd, mu_J, sigma_J, steps = steps, paths = paths, Delta_t = Delta_t)
        sim_fig = px.line(S_t, title = f'Simulación de los precios futuros de {stock_symbol}')
        sim_fig.update_layout(xaxis_title='Tiempo Simulacion (Dias)', yaxis_title='Precio Accion ($)', showlegend=False)

        # Grafico del VaR
        retornos = (S_t[-1,:] - S0)/S0
        var = calculate_var(retornos, alpha)
        var_fig = go.Figure()
        var_fig.add_trace(go.Histogram(x = retornos, nbinsx = 500, showlegend = False))
        var_fig.update_layout(title=f'VaR a {steps} días con un nivel de confianza del {1-alpha:.2%}: {var:.2%}',xaxis=dict(title = "Retornos", tickformat = ".0%",), yaxis_title = "Frecuencia")

        # Grafico valoracion opciones
        strikes = S0*np.arange(0.25, 3, 0.20)
        V = mc_option_valuation(S0, strikes, steps, r, mu, sigma, lambd, mu_J, sigma_J, paths, option_type)
        option_fig = px.line(x = strikes, y = V, title = f'Precio de una opción {option_type} sobre {stock_symbol} con vencimiento en {steps} días')
        option_fig.update_layout(xaxis_title = 'Strikes ($)', yaxis_title = 'Precio de la Opcion ($)', showlegend = False)
    
    return stock_fig, sim_fig, var_fig, option_fig

# Mantenemos el dashboard en constante actualizacion (al realizar cambios)
if __name__ == '__main__':
    app.run_server(debug=True)

