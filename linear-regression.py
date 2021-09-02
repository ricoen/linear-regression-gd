import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Relationship between power (kW) and Torque (Nm) with Motor Speed 2000 rpm
# https://www.engineeringtoolbox.com/electrical-motors-hp-torque-rpm-d_1503.html
x = np.array([1.1, 1.5, 2.2, 3.7, 5.6])
y = np.array([5.3, 7.1, 10.7, 18, 27])

def gradient_descent(epochs, alpha):
    m = 0
    c = 0
    n = len(x)

    for _ in range(epochs):
        Y_pred = m*x + c
        
        D_m = (-2/n)*sum(x*(y - Y_pred))
        D_c = (-2/n)*sum(y - Y_pred)
        
        m = m - alpha * D_m
        c = c - alpha * D_c
    
    return [m, c]
        
def predict(x_train):
    [m, c] = gradient_descent(1000, 0.01)
    print(m, c)
    Y_predict = m*x_train + c
    return Y_predict

y_prediction = predict(x)
y_prediction = np.round(y_prediction, 2)

print(y_prediction)

fig1 = px.scatter(x=x, y=y, color_discrete_sequence=['green'])
fig = go.Figure(data=fig1.data)
fig.add_trace(go.Scatter(x=x, y=y_prediction, name='Linear Regression'))
fig.show()
