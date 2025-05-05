import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.random.set_seed(21)
np.random.seed(21)
#entradas
x = np.array([
    [15, 0],
    [79, 0],
    [80, 1],
    [75, 0],
    [100, 1],
    [29, 1],
    [46, 1],
    [82, 0],
    [97, 1],
    [12, 0],
    [79, 1]
], dtype=float)

y = np.array([[0], [0], [1], [0], [1], [0], [0], [0], [1], [0], [0]], dtype=int)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=1, activation='sigmoid') # neurona de salida OR
])

#compilacion del modelo
lr = 0.1 #learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
    )

print('Model summary:')
history = model.fit(x, y, epochs=1000, verbose=False)
print('Modelo entrenado.')

# Predicciones
result = model.predict(x, verbose=0)
for i, res in enumerate(result):
    print(f"Entrada: {x[i]}, Predicción: {res[0]:.4f}, Esperado: {y[i][0]}, error: {abs(res[0] - y[i][0]):.4f}%")
