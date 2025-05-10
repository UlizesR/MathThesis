import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

def drdphi(phi, r, b, M=1):
    # Common integrand (signed via phi range)
    return np.sign(phi) * np.sqrt(np.maximum(r**4/b**2 - r**2 + 2*M*r, 0))

def compute_trajectory(b, phi_max=4, M=1, r_init=100):
    # inbound: phi from -phi_max to 0
    sol = solve_ivp(drdphi, [-phi_max, phi_max], [r_init],
                    args=(b, M), max_step=0.1)
    phi = sol.t
    r   = sol.y[0]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

# Parameters
b_values = [10, 7, 5.5, 4]
r_ps = 3  # photon sphere (3 GM with M=1)

# Build Plotly figure
fig = go.Figure()

# Null geodesic traces
for b in b_values:
    x, y = compute_trajectory(b)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name=f'b = {b}'
    ))

# Photon sphere
theta = np.linspace(0, 2*np.pi, 300)
fig.add_trace(go.Scatter(
    x=r_ps * np.cos(theta),
    y=r_ps * np.sin(theta),
    mode='lines',
    name='Photon sphere (r=3 M)',
    line=dict(dash='dash')
))

# Layout styling
fig.update_layout(
    title='Null Geodesics around a Schwarzschild Black Hole (M=1)',
    xaxis_title='x',
    yaxis_title='y',
    width=600, height=600
)

# Show with scroll-wheel zoom enabled
config = {'scrollZoom': True}
fig.show(config=config)
