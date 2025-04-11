import numpy as np

# Datos de entrada
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# datos de salida
y = np.array([0, 0, 0, 1])  

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
        xi = X[i]
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
