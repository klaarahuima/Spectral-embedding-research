import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

# Import modern font
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'
]

# 1. ADDED VIEWPORT META TAG FOR MOBILE DEVICES
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)
app.title = "Circulant Graph Embedding"
server = app.server

# Default parameters
DEFAULT_N = 30
DEFAULT_K = 4
DEFAULT_Q1_EXTRA = 0
DEFAULT_P = 0.0

# Reusable UI Styles
CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
    'padding': '24px',
    'marginBottom': '24px'
}

TEXT_STYLE = {
    'color': '#4a5568',
    'fontSize': '15px',
    'lineHeight': '1.6'
}

LABEL_STYLE = {
    'fontWeight': '600',
    'color': '#2d3748',
    'marginBottom': '8px',
    'display': 'block',
    'fontSize': '14px'
}

app.layout = html.Div([
    
    # ---------------- HEADER ----------------
    html.Div([
        html.H1("Spectral Embedding of Random Circulant Graphs", 
                style={'textAlign': 'center', 'color': '#1a202c', 'fontWeight': '700', 'marginBottom': '10px'}),
        html.P([
            "An interactive demo | ",
            html.A("View project on GitHub", 
                   href="YOUR_GITHUB_LINK_HERE",  # <-- Replace with your actual GitHub URL
                   target="_blank",
                   style={'color': '#3182ce', 'textDecoration': 'none', 'fontWeight': '600'})
        ], style={'textAlign': 'center', 'color': '#718096', 'fontSize': '18px', 'marginTop': '0'})
    ], style={'padding': '40px 20px 20px'}),

    # ---------------- INTRO & CONTROLS ----------------
    html.Div([
        # Introduction Paragraph
        html.Div([
            html.H3("Research Overview", style={'marginTop': '0', 'color': '#2d3748'}),
            html.P("We can embed graphs in 2 dimensions by using of their 2nd and 3rd Laplacian eigenvectors. "
                   "The embedding of a circulant graph forms a smooth circle where the vertices perfectly retain their original cyclic ordering. "
                   "This research studied the effect of random perturbations on the Laplacian embedding of circulant graphs. ",
                   style=TEXT_STYLE),

            html.P("Below, you can explore these effects interactively by introducing a denser first quadrant (the vertices in the first quadrant have more neighbors than the rest of the graph) or varying the probability that we remove any given edge from the graph. "
                               "As the layout is perturbed, the original circular ordering of the vertices is lost.", 
                               style=TEXT_STYLE),
        ], style={**CARD_STYLE, 'borderLeft': '5px solid #3182ce'}),

        # Control Panel (UPDATED FOR RESPONSIVENESS)
        html.Div([
            html.Div([
                html.Label("Graph size (n)", style=LABEL_STYLE),
                dcc.Slider(id='n-slider', min=10, max=50, step=5, value=DEFAULT_N, 
                           marks={i: str(i) for i in range(10, 51, 10)},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={'flex': '1 1 200px', 'minWidth': '200px'}),
            
            html.Div([
                html.Label("Number of neighbors", style=LABEL_STYLE),
                dcc.Slider(id='k-slider', min=1, max=8, step=1, value=DEFAULT_K, 
                           marks={i: str(i) for i in range(1, 9)}),
            ], style={'flex': '1 1 200px', 'minWidth': '200px'}),

            html.Div([
                html.Label("Additional neighbors in Q1", style=LABEL_STYLE),
                dcc.Slider(id='q1-extra-slider', min=0, max=8, step=1, value=DEFAULT_Q1_EXTRA, 
                           marks={i: str(i) for i in range(0, 9, 1)}),
            ], style={'flex': '1 1 200px', 'minWidth': '200px'}),
            
            html.Div([
                html.Label("Edge removal probability", style=LABEL_STYLE),
                dcc.Slider(id='p-slider', min=0.0, max=1.0, step=0.05, value=DEFAULT_P, 
                           marks={i/10: str(i/10) for i in range(0, 11, 2)}),
            ], style={'flex': '1 1 200px', 'minWidth': '200px'})
            
        ], style={**CARD_STYLE, 'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px'})
    ]),

    # ---------------- VISUALIZATIONS (UPDATED FOR RESPONSIVENESS) ----------------
    html.Div([
        # Left Side: Network Topology
        # Left Side: Network Topology
        html.Div([
            html.Div([
                html.H3("Graph structure", style={'marginTop': '0', 'color': '#2d3748', 'textAlign': 'center'}),
                
                # --- UPDATE THIS LINE BELOW ---
                dcc.Graph(id='static-graph', config={'staticPlot': True}, style={'width': '100%', 'aspectRatio': '1', 'maxHeight': '550px', 'margin': 'auto'}),
                
                html.P("The generated graph visualized on a unit circle. The first quadrant (Q1) is highlighted with orange diamond markers. "
                       "If additional neighbors are added, there'll be a higher concentration of edges in this region.", 
                       style={**TEXT_STYLE, 'marginTop': '15px'})
            ], style=CARD_STYLE)
        ], style={'flex': '1 1 45%', 'minWidth': '300px'}),
        
        # Right Side: Embeddings & Eigenvectors
        html.Div([
            html.Div([
                html.H3("2D Spectral Embedding", style={'marginTop': '0', 'color': '#2d3748', 'textAlign': 'center'}),
                dcc.Graph(id='embedding-plot', style={'height': '380px'}),
                html.P("Plotted using the 2nd and 3rd Laplacian eigenvectors. The node colors follow a rainbow gradient based on their original cyclic order. "
                       "Adding dense regions or randomly removing edges modifies the layout, losing the perfect circular structure. ",
                       style ={**TEXT_STYLE, 'marginTop' : '15px'}),
                html.P(
                       "If the layout is no longer circular at all, your graph is likely disconnected. Try changing your parameters.", 
                       style={**TEXT_STYLE, 'marginTop': '15px'})
            ], style=CARD_STYLE),

            html.Div([
                html.H3("1D Eigenvector Projection", style={'marginTop': '0', 'color': '#2d3748', 'textAlign': 'center'}),
                dcc.Graph(id='eigenvectors-plot', style={'height': '220px'}),
                html.P("The coordinate values of the 2nd (blue) and 3rd (red) eigenvectors plotted against the original vertex indices. "
                       "In an ideal circulant graph, these form perfect sinusoidal waves. Randomly removing edges warps the underlying sinusoidal structure, "
                       "while creating denser regions results in eigenvectors resembling combinations of two sinusoidal waves (with different parameters) patched together.", 
                       style={**TEXT_STYLE, 'marginTop': '15px'}) 
            ], style=CARD_STYLE)
        ], style={'flex': '1 1 45%', 'minWidth': '300px'})
        
    ], style={'display': 'flex', 'flexWrap': 'wrap-reverse', 'gap': '20px'})
    
], style={'fontFamily': "'Inter', sans-serif", 'maxWidth': '1350px', 'margin': 'auto', 'backgroundColor': '#f7fafc', 'padding': '20px'})


@app.callback(
    [Output('static-graph', 'figure'),
     Output('embedding-plot', 'figure'),
     Output('eigenvectors-plot', 'figure')],
    [Input('n-slider', 'value'),
     Input('k-slider', 'value'),
     Input('q1-extra-slider', 'value'),
     Input('p-slider', 'value')]
)
def update_dashboard(n, k, q1_extra, p):
    
    # 1. Generate Graph
    base_edges = set()
    q1_end = max(1, n // 4)
    
    for i in range(n):
        current_k = k + q1_extra if i < q1_end else k
        for j in range(1, current_k + 1):
            target_fwd = (i + j) % n
            base_edges.add((min(i, target_fwd), max(i, target_fwd)))
            target_bwd = (i - j) % n
            base_edges.add((min(i, target_bwd), max(i, target_bwd)))
            
    edges = []
    for u, v in base_edges:
        if np.random.rand() >= p:
            edges.append([u, v])

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    # 2. Compute Eigenvectors
    try:
        L = nx.laplacian_matrix(G).astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L.todense())
        vec2 = np.asarray(eigenvectors[:, 1]).flatten()
        vec3 = np.asarray(eigenvectors[:, 2]).flatten()
    except Exception:
        vec2, vec3 = np.zeros(n), np.zeros(n)

    # 3. Static Graph Figure
    static_fig = go.Figure()
    pos_x, pos_y = [], []
    radius = 1
    
    for i in range(n):
        theta = 2 * np.pi * i / n
        pos_x.append(radius * np.cos(theta))
        pos_y.append(radius * np.sin(theta))
        
    edge_x, edge_y = [], []
    for u, v in edges:
        edge_x.extend([pos_x[u], pos_x[v], None])
        edge_y.extend([pos_y[u], pos_y[v], None])
        
    static_fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#e2e8f0', width=1), hoverinfo='none'))
    static_fig.add_trace(go.Scatter(
        x=pos_x, y=pos_y, mode='markers',
        marker=dict(
            size=[16 if i < q1_end else 10 for i in range(n)],
            symbol=['diamond' if i < q1_end else 'circle' for i in range(n)],
            color=['#ed8936' if i < q1_end else '#4299e1' for i in range(n)],
            line=dict(width=1, color='#2d3748')
        ),
        text=[f"Node {i}" for i in range(n)], hoverinfo='text'
    ))
    static_fig.update_layout(
        showlegend=False, 
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), # <-- Added scale anchor here
        template='plotly_white', margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )

    # 4. Embedding Figure
    marker_symbols = ['diamond' if i < q1_end else 'circle' for i in range(n)]

    embedding_fig = go.Figure()
    embedding_fig.add_trace(go.Scatter(
        x=vec2, y=vec3, mode='markers',
        marker=dict(
            symbol=marker_symbols, color=list(range(n)),
            colorscale='Rainbow', showscale=True, 
            colorbar=dict(
                title=dict(text='Node Index', font=dict(family='Inter')), 
                thickness=6,
                orientation='h', 
                y=-0.3           
            ),
            line=dict(color='#2d3748')
        ),
        text=[f"Node {i}" for i in range(n)],
        hovertemplate="<b>Node %{text}</b><br>2nd Eig: %{x:.4f}<br>3rd Eig: %{y:.4f}<extra></extra>"
    ))
    embedding_fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=80), # Slightly reduced bottom margin
        template='plotly_white',
        font=dict(family='Inter', color='#4a5568', size=11), # Smaller global text
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title=dict(text='3rd Eigenvector', font=dict(size=12), standoff=1), # Smaller title, pulled closer to axis
            scaleanchor="x", scaleratio=1, gridcolor='#edf2f7'
        ),
        xaxis=dict(
            title=dict(text='2nd Eigenvector', font=dict(size=12), standoff=5),
            gridcolor='#edf2f7'
        )
    )
                                
    # 5. 1D Eigenvector Figure
    eigen_fig = go.Figure()
    eigen_fig.add_trace(go.Scatter(x=list(range(n)), y=vec2, mode='lines+markers', name='2nd Eig', line=dict(color='#3182ce', width=2), marker=dict(size=5)))
    eigen_fig.add_trace(go.Scatter(x=list(range(n)), y=vec3, mode='lines+markers', name='3rd Eig', line=dict(color='#e53e3e', width=2), marker=dict(size=5)))
    eigen_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30), 
        template='plotly_white',
        font=dict(family='Inter', color='#4a5568', size=11),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title=dict(text='Coordinate', font=dict(size=12), standoff=1),
            gridcolor='#edf2f7'
        ),
        xaxis=dict(
            title=dict(text='Node Index', font=dict(size=12), standoff=5),
            gridcolor='#edf2f7'
        )
    )
                            
    return static_fig, embedding_fig, eigen_fig

if __name__ == '__main__':
    app.run(debug=True)