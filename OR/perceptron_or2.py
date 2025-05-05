
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.random.set_seed(1234)
np.random.seed(4321)
# Entradas
x = np.array([
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
    [45, 50],

], dtype=float)

#salidas
y = np.array([[0], [1], [1], [1], [0], [1], [1], [1], [0], [1], [0]], dtype=float)

#########definicion del modelo##############
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=1, activation='sigmoid') #  neurona de salida OR
])

#compilacion del modelo
lr = 0.001 #learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
    )


print('Model summary:')
history = model.fit(x, y, epochs=5000, verbose=False)
print('Modelo entrenado.')

# Predicciones
result = model.predict(x, verbose=0)
for i, res in enumerate(result):
    print(f"Entrada: {x[i]}, Predicción: {res[0]:.4f}, Esperado: {y[i][0]}, error: {abs(res[0] - y[i][0]):.4f}%")


# Entradas y etiquetas
x1 = x[:, 0]
x2 = x[:, 1]
labels = y.flatten()

# Obtener pesos y bias del modelo (última capa, salida)
pesos, sesgos = model.layers[-1].get_weights()
w1, w2 = pesos[0][0], pesos[1][0]  # Pesos de la neurona de salida
# Bias (sesgo) de la neurona de salida
b = sesgos[0]

print(f"w1: {w1:.2f}, w2: {w2:.2f}, b: {b:.2f}")

# # Crear figura
# plt.figure(figsize=(7, 6))

# # Graficar puntos de datos con sus clases
# for i in range(len(labels)):
#     color = 'blue' if labels[i] == 0 else 'red'
#     plt.scatter(x1[i], x2[i], c=color, edgecolor='k', s=100)

# # Crear línea de decisión: w1 * x + w2 * y + b = 0  →  y = -(w1/w2)x - b/w2
# x_vals = np.linspace(-0.1, 1.1, 100)
# if w2 != 0:
#     decision_boundary = -(w1 / w2) * x_vals - (b / w2)
#     plt.plot(x_vals, decision_boundary, 'k--', label='Frontera de decisión')
# else:
#     # Caso raro: w2 = 0 → frontera vertical
#     x_const = -b / w1
#     plt.axvline(x=x_const, linestyle='--', color='k', label='Frontera de decisión')

# # Configuración de la gráfica
# plt.title('Perceptrón como clasificador lineal')
# plt.xlabel('Entrada 1: Temperatura')
# plt.ylabel('Entrada 2: Humo')
# plt.legend()
# plt.grid(True)
# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 1.1)
# plt.show()