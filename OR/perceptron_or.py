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

# Inicialización de pesos y bias
w = np.random.rand(2)
b = np.random.rand()
lr = 0.1  # tasa de aprendizaje
epochs = 20

# Función de activación escalón (step)
def step(x):
    return 1 if x >= 0 else 0

# Entrenamiento
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
        x1 = 0 if X[i][0] < 50 else 1
        x2 = 0 if X[i][1] < 60 else 1
        xi = np.array([x1, x2])
        y_pred = step(np.dot(w, xi) + b)
        error = y[i] - y_pred
        
        # Actualización de pesos y bias
        w += lr * error * xi
        b += lr * error
        
        print(f"Entrada: {xi}, Predicho: {y_pred}, Esperado: {y[i]}, Error: {error}")
    print("-" * 40)

# Prueba del perceptrón entrenado
print("Resultados finales:")
for xi in X:
    result = step(np.dot(w, xi) + b)
    print(f"Entrada: {xi}, Salida: {result}")
