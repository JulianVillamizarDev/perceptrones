import numpy as np

# Datos de entrada y salidas esperadas para OR
X = np.array([
    [20, 35],
    [55, 40],
    [10, 65],
    [70, 80],
    [49, 59],
    [51, 61],
    [50, 30],
    [30, 70],
    [0, 0],
    [50, 60],
    [45, 50]
])

# salida esperada (OR)
y = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0])
y_predict = np.zeros_like(y)  

# Inicialización de pesos y bias
w = np.random.rand(2)
b = np.random.rand()
lr = 0.1  # tasa de aprendizaje
epochs = 50

# Función de activación escalón (step)
def step(x):
    return 1 if x >= 0 else 0

# Entrenamiento
for epoch in range(epochs):
    for i in range(len(X)):
        # Normalización de entradas
        x1 = 0 if X[i][0] < 50 else 1
        x2 = 0 if X[i][1] < 60 else 1
        xi = np.array([x1, x2])

        # Predicción con las entradas normalizadas
        y_pred = step(np.dot(w, xi) + b)
        error = y[i] - y_pred
        
        # Actualización de pesos y bias
        w += lr * error * xi
        b += lr * error
        
        y_predict[i] = y_pred  # Guardar la predicción

# Prueba del perceptrón entrenado
print("Resultados finales:")
for i in range(X.shape[0]):
    print(f"Entrada: {X[i]}, Salida: {y_predict[i]}, Esperado: {y[i]}")
