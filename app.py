import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- 1. Tạo Dữ liệu Giả lập ---
def generate_data(n_samples=100, round_num=1, random_state=42):
    fixed_centers = [[-3, -3], [3, 3]]    
    dynamic_std = 1.0 + (round_num * 0.25)
    X, y = make_blobs(n_samples=n_samples, centers=fixed_centers, 
                      cluster_std=dynamic_std, random_state=random_state)
    return X, y

# --- 2. Dashboard Dash ---
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H2("Thử Thách Phân Loại: Đấu Với Máy (SVM)", style={'text-align': 'center', 'margin': '10px 0'}),
    html.Div([
        html.Div([
            html.Div([
                html.H4("Thống kê trò chơi", style={'margin': '0 0 5px 0'}),
                html.P(id='game-stats-text', style={'font-size': '15px', 'color': '#2c3e50', 'margin': '2px 0'}),
                html.P(id='user-accuracy-text', style={'font-weight': 'bold', 'color': '#e67e22', 'margin': '2px 0', 'font-size': '13px'}),
                html.P(id='svm-accuracy-text', style={'font-weight': 'bold', 'color': '#3498db', 'font-size': '13px', 'margin': '2px 0'}),
            ], style={'flex': '1'}),
            html.Div([
                html.Div([
                    html.H5("Biểu đồ hiệu suất", style={'margin': '0 0 5px 0', 'text-align': 'center'}),
                    dcc.Graph(
                        id='score-history-bar',
                        config={'displayModeBar': False},
                        style={'height': '100px'}
                    )
                ], id='history-container', style={'display': 'none'})
            ], style={'flex': '2', 'padding': '0 10px'}),
            html.Div([
                html.Button('Hiện/Ẩn SVM', id='show-svm-btn', n_clicks=0, 
                            style={'margin': '2px', 'padding': '5px', 'font-size': '12px'}),
                html.Button('Chốt & Sang Vòng', id='next-round-btn', n_clicks=0, 
                            style={'margin': '2px', 'padding': '8px', 'background-color': '#27ae60', 'color': 'white', 'border': 'none', 'font-weight': 'bold', 'font-size': '12px'}),
                html.Button('Chơi lại', id='reset-game-btn', n_clicks=0, 
                            style={'margin': '2px', 'padding': '3px', 'font-size': '10px', 'background-color': '#ecf0f1', 'border': '1px solid #bdc3c7'}),
            ], style={'flex': '1', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}),
        ], style={
            'display': 'flex', 'width': '90%', 'margin': 'auto', 'padding': '10px 20px', 
            'border': '1px solid #34495e', 'border-radius': '10px', 'background-color': '#f8f9f9',
            'align-items': 'center', 'max-height': '130px'
        }),
        dcc.ConfirmDialog(id='game-over-dialog', message=''),
        html.Div([
            dcc.Graph(
                id='scatter-plot',
                config={
                    'edits': {'shapePosition': True},
                    'modeBarButtonsToAdd': ['drawline'],
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'displayModeBar': True
                },
                style={'height': '70vh'}
            ),
        ], style={'width': '95%', 'margin': 'auto'}),
    ]),
    dcc.Store(id='data-store'),
    dcc.Store(id='svm-store'),
    dcc.Store(id='game-state-store', data={
        'score': 0.0, 
        'round': 1, 
        'attempts': 0, 
        'best_acc_in_round': 0.0,
        'history': []
    }),
], style={'font-family': 'Segoe UI, Arial, sans-serif', 'overflow': 'hidden'})

# --- 3. Callbacks ---
@app.callback(
    [Output('data-store', 'data'),
     Output('svm-store', 'data'),
     Output('svm-accuracy-text', 'children')],
    [Input('next-round-btn', 'n_clicks'),
     Input('reset-game-btn', 'n_clicks')],
    [State('game-state-store', 'data')]
)
def update_dataset(n_next, n_reset, game_state):
    current_round = game_state.get('round', 1)
    seed = np.random.randint(1, 10000)
    X, y = generate_data(round_num=current_round, random_state=seed)
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)
    w, b = svm.coef_[0], svm.intercept_[0]
    denom = w[1] if abs(w[1]) > 1e-5 else 1e-5
    slope, intercept = -w[0] / denom, -b / denom
    acc_svm = accuracy_score(y, svm.predict(X))
    return {'X': X.tolist(), 'y': y.tolist()}, \
           {'slope': slope, 'intercept': intercept, 'acc': acc_svm}, \
           f"Máy (SVM): {acc_svm*100:.2f}%"
@app.callback(
    [Output('game-state-store', 'data'),
     Output('game-stats-text', 'children'),
     Output('game-over-dialog', 'displayed'),
     Output('game-over-dialog', 'message'),
     Output('score-history-bar', 'figure'),
     Output('history-container', 'style')],
    [Input('scatter-plot', 'relayoutData'),
     Input('next-round-btn', 'n_clicks'),
     Input('reset-game-btn', 'n_clicks')],
    [State('game-state-store', 'data'),
     State('data-store', 'data'),
     State('svm-store', 'data')]
)
def manage_game(relayout, n_next, n_reset, state, data, svm_info):
    triggered = ctx.triggered_id    
    if triggered == 'reset-game-btn':
        state = {'score': 0.0, 'round': 1, 'attempts': 0, 'best_acc_in_round': 0.0, 'history': [], 'game_over': False}
        return state, "Điểm: 0.0 | Vòng: 1/10", False, "", go.Figure(), {'display': 'none'}
    if state.get('game_over'):
        return dash.no_update
    if triggered == 'scatter-plot' and relayout and 'shapes' in relayout:
        user_shapes = [s for s in relayout['shapes'] if s.get('type') == 'line']
        if user_shapes and data and state['attempts'] < 3:
            line = user_shapes[-1]
            X, y = np.array(data['X']), np.array(data['y'])
            x0, y0, x1, y1 = line['x0'], line['y0'], line['x1'], line['y1']
            if x1 != x0:
                m = (y1 - y0) / (x1 - x0)
                c = y0 - m * x0
                acc = max(accuracy_score(y, (X[:, 1] > (m * X[:, 0] + c)).astype(int)), 
                          1 - accuracy_score(y, (X[:, 1] > (m * X[:, 0] + c)).astype(int)))
                state['attempts'] += 1
                state['best_acc_in_round'] = max(state['best_acc_in_round'], acc)
    if triggered == 'next-round-btn':
        round_score = (state['best_acc_in_round'] - svm_info['acc']) * 100
        state['score'] += round_score
        state['history'].append(round_score)       
        if state['score'] < 0 or state['round'] >= 10:
            state['game_over'] = True
            final_msg = f"KẾT THÚC! Tổng điểm: {state['score']:.1f}. Xem biểu đồ bên dưới."
            fig_bar = go.Figure(go.Bar(
                x=[f"V{i+1}" for i in range(len(state['history']))],
                y=state['history'],
                marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in state['history']]
            ))
            fig_bar.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=100, template="simple_white")
            
            return state, f"TỔNG ĐIỂM: {state['score']:.1f}", True, final_msg, fig_bar, {'display': 'block'}
        state['round'] += 1
        state['attempts'] = 0
        state['best_acc_in_round'] = 0.0
    msg = f"Điểm: {state['score']:.1f} | Vòng: {state['round']}/10 | Thử: {state['attempts']}/3"
    return state, msg, False, "", dash.no_update, {'display': 'none'}
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('user-accuracy-text', 'children')],
    [Input('data-store', 'data'),
     Input('scatter-plot', 'relayoutData'),
     Input('show-svm-btn', 'n_clicks')],
    [State('svm-store', 'data'),
     State('game-state-store', 'data')]
)
def update_graph(data, relayout, svm_clicks, svm_info, game_state):
    if not data: return go.Figure(), ""
    X, y = np.array(data['X']), np.array(data['y'])
    shapes = []   
    if game_state.get('game_over'):
        current_dragmode = False
        user_acc_msg = "TRÒ CHƠI KẾT THÚC. Hãy nhấn 'Chơi lại' để bắt đầu ván mới."
    else:
        current_dragmode = 'drawline' if game_state['attempts'] < 3 else 'pan'
        user_acc_msg = "Vòng mới! Hãy vẽ đường phân chia."
        if game_state['attempts'] >= 3:
            user_acc_msg = f"Hết lượt! Best Acc: {game_state['best_acc_in_round']*100:.2f}%"
    if ctx.triggered_id == 'scatter-plot' and relayout and 'shapes' in relayout:
        user_shapes = [s for s in relayout['shapes'] if s.get('type') == 'line']
        if user_shapes:
            line = user_shapes[-1]
            shapes.append(line)
            x0, y0, x1, y1 = line['x0'], line['y0'], line['x1'], line['y1']
            if x1 != x0:
                m, c = (y1-y0)/(x1-x0), y0 - ((y1-y0)/(x1-x0))*x0
                acc = max(accuracy_score(y, (X[:, 1] > (m*X[:, 0]+c)).astype(int)), 
                          1 - accuracy_score(y, (X[:, 1] > (m*X[:, 0]+c)).astype(int)))
                if not game_state.get('game_over') and game_state['attempts'] < 3:
                    user_acc_msg = f"Lượt này: {acc*100:.2f}% (Best: {max(game_state['best_acc_in_round'], acc)*100:.2f}%)"
    if svm_clicks % 2 == 1:
        shapes.append({'type': 'line', 'x0': X[:, 0].min(), 'y0': svm_info['slope']*X[:, 0].min()+svm_info['intercept'],
                       'x1': X[:, 0].max(), 'y1': svm_info['slope']*X[:, 0].max()+svm_info['intercept'],
                       'line': {'color': 'black', 'width': 2, 'dash': 'dot'}})
    fig = go.Figure()
    for label, color, name in [(0, '#0000FF', 'A'), (1, '#FF0000', 'B')]:
        mask = y == label
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=name,
                                 marker=dict(size=10, color=color, opacity=0.7)))
    fig.update_layout(
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), 
        shapes=shapes, 
        dragmode=current_dragmode,
        uirevision=str(data), 
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig, user_acc_msg

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
