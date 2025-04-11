import numpy as np

# Función de activación y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Datos de entrada y salida esperada (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Inicialización de pesos y biases
np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

# Pesos
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hiperparámetros
lr = 0.1
epochs = 10000

# Entrenamiento
for epoch in range(epochs):
    # FORWARD
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # salida final

    # BACKPROPAGATION
    error = y - a2
    d_a2 = error * sigmoid_derivative(z2)
    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(z1)

    # Actualización de pesos y bias
    W2 += lr * np.dot(a1.T, d_a2)
    b2 += lr * np.sum(d_a2, axis=0, keepdims=True)
    W1 += lr * np.dot(X.T, d_a1)
    b1 += lr * np.sum(d_a1, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}, Loss: {loss}")

# Prueba del modelo entrenado
print("\nResultados finales:")
for i in range(len(X)):
    z1 = np.dot(X[i], W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    print(f"Entrada: {X[i]} → Predicción: {a2[0]:.4f} → Clasificación: {round(a2[0])}")
