
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#entradas
x = np.array([
    [20, 35],
    [55, 40],
    [10, 65],
    [70, 80],
    [49, 59],
    [51, 61],
    [50, 30],
    [30, 70],
    [0, 0]
], dtype=float)

#salidas
y = np.array([[0], [1], [1], [1], [0], [1], [1], [1], [0]], dtype=int)


#########definicion del modelo##############
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_shape=[2]), #neurona de conversion a binario
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
history = model.fit(x, y, epochs=500, verbose=False)
print('Modelo entrenado.')

# Predicciones
result = model.predict(x, verbose=0)
for i, res in enumerate(result):
    print(f"Entrada: {x[i]}, Predicción: {res[0]:.4f}, Esperado: {y[i][0]}")
