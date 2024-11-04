import matplotlib.pyplot as plt

datasetx = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
datasety = [8.3, 14.5, 9, 21.9, 16, 30.1, 36.05, 42.9, 52.4, 38]

def derivadas(datasetx, datasety, a, b, c):
    derivada_a = 0
    derivada_b = 0
    derivada_c = 0
    for i in range(len(datasetx)):
        erro = a * datasetx[i]**2 + b * datasetx[i] + c - datasety[i]
        derivada_a = derivada_a + 2 * erro * datasetx[i]**2
        derivada_b = derivada_b + 2 * erro * datasetx[i]
        derivada_c = derivada_c + 2 * erro
    return derivada_a, derivada_b, derivada_c

def distancia(x_posterior, y_posterior, z_posterior, x_anterior, y_anterior, z_anterior):
    return ((x_posterior - x_anterior)**2 + (y_posterior - y_anterior)**2 + (z_posterior - z_anterior)**2)**0.5

def gradiente_descendente(a, b, c, tolerancia, learning_rate):
    x_anterior = a
    y_anterior = b
    z_anterior = c

    i = 0
    while True:
        derivada = derivadas(datasetx, datasety, x_anterior, y_anterior, z_anterior)
        x_posterior = x_anterior - learning_rate * derivada[0]
        y_posterior = y_anterior - learning_rate * derivada[1]
        z_posterior = z_anterior - learning_rate * derivada[2]
        
        dist = distancia(x_posterior, y_posterior, z_posterior, x_anterior, y_anterior, z_anterior)
        i+=1

        if dist > tolerancia:
            x_anterior = x_posterior
            y_anterior = y_posterior
            z_anterior = z_posterior
        else:
            break
    return i, x_anterior, y_anterior, z_anterior


iteracoes, a_final, b_final, c_final = gradiente_descendente(2, 4, 8, 10**(-16), 0.00001)
print(f"Número de iterações: {iteracoes}, valor final de A: {a_final}, B: {b_final} e C: {c_final}")


x_grafico = [i * 0.1 for i in range(71)]
y_grafico = [a_final * x**2 + b_final * x + c_final for x in x_grafico]

plt.figure(figsize=(10, 6))
plt.scatter(datasetx, datasety, color='blue', label='Pontos do dataset')
plt.plot(x_grafico, y_grafico, label='Curva quadrática ajustada', color='red')
plt.title('Equação quadrática usando gradiente descendente')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()