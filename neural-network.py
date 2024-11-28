import math as math
import matplotlib.pyplot as plt
import pandas as pd

# Função para calcular a Acurácia
def accuracy(dtx, dty, a, b, c, d, k, l):
    acertos = 0
    for i in range(len(dtx)):
        x = dtx[i]
        y_real = dty[i]
        u = sigmoid(a * x + b)
        w = sigmoid(c * u + d)
        z = sigmoid(k * w + l)
        y_pred = 1 if z >= 0.5 else 0  # Previsão binária (limiar 0.5)
        if y_pred == y_real:
            acertos += 1
    return acertos / len(dtx)

# Função para calcular o F1 Score
def f1_score(dtx, dty, a, b, c, d, k, l):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for i in range(len(dtx)):
        x = dtx[i]
        y_real = dty[i]
        u = sigmoid(a * x + b)
        w = sigmoid(c * u + d)
        z = sigmoid(k * w + l)
        y_pred = 1 if z >= 0.5 else 0  # Previsão binária (limiar 0.5)
        
        if y_pred == 1 and y_real == 1:
            tp += 1
        elif y_pred == 1 and y_real == 0:
            fp += 1
        elif y_pred == 0 and y_real == 1:
            fn += 1
    
    # Evitar divisão por zero
    if tp + fp == 0 or tp + fn == 0:
        return 0
    
    precision = tp / (tp + fp)  # Precisão
    recall = tp / (tp + fn)     # Recall
    return 2 * (precision * recall) / (precision + recall)  # F1 Score

def sigmoid(x):
    return 1/(1+math.exp(-x))

def derivada_da_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#normaliza os dados para diminuar a escala e torná-los comparáveis
def normalizar(dado):
    media = sum(dado)/len(dado)
    variancia = sum((x - media)**2 for x in dado) / len(dado)
    raiz = variancia**0.5
    dado_normalizado = [(x - media) / raiz for x in dado]
    return dado_normalizado

#calculando as derivadas de cada coeficiente da rede neural
def derivadas(datasetx, datasety, a, b, c, d, k, l):
    derivada_a = 0
    derivada_b = 0
    derivada_c = 0
    derivada_d = 0
    derivada_k = 0
    derivada_l = 0
    for i in range(len(datasetx)):
        u = sigmoid(a * datasetx[i] + b)
        w = sigmoid(c * u + d)
        z = sigmoid(k * w + l)
        erro = datasety[i] - z

        derivada_z = -2 * erro * derivada_da_sigmoid(k * w + l)                # derivada em relação a z
        derivada_w = derivada_z * k * derivada_da_sigmoid(c * u + d)           # derivada em relação a w
        derivada_u = derivada_w * c * derivada_da_sigmoid(a * datasetx[i] + b) # derivada em relação a u

        # Parâmetros
        derivada_a += derivada_u * datasetx[i] #derivada em relação a A
        derivada_b += derivada_u               #derivada em relação a B
        derivada_c += derivada_w * u           #derivada em relação a C
        derivada_d += derivada_w               #derivada em relação a D
        derivada_k += derivada_z * w           #derivada em relação a K
        derivada_l += derivada_z               #derivada em relação a L
    return derivada_a, derivada_b, derivada_c, derivada_d, derivada_k, derivada_l

def distancia(a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior,
              a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior):
    return ((a_posterior - a_anterior)**2 +
            (b_posterior - b_anterior)**2 +
            (c_posterior - c_anterior)**2 +
            (d_posterior - d_anterior)**2 +
            (k_posterior - k_anterior)**2 +
            (l_posterior - l_anterior)**2)**0.5

def gradiente_descendente(a, b, c, d, k, l, tolerancia, learning_rate):
    a_anterior = a
    b_anterior = b
    c_anterior = c
    d_anterior = d
    k_anterior = k
    l_anterior = l
    a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior = [0.1] * 6

    i = 0
    while True:
        derivada = derivadas(datasetx, datasety, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior)
        a_posterior = a_anterior - learning_rate * derivada[0]
        b_posterior = b_anterior - learning_rate * derivada[1]
        c_posterior = c_anterior - learning_rate * derivada[2]
        d_posterior = d_anterior - learning_rate * derivada[3]
        k_posterior = k_anterior - learning_rate * derivada[4]
        l_posterior = l_anterior - learning_rate * derivada[5]
        
        dist = distancia(a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior)
        i += 1

        if dist > tolerancia:
            a_anterior = a_posterior
            b_anterior = b_posterior
            c_anterior = c_posterior
            d_anterior = d_posterior
            k_anterior = k_posterior
            l_anterior = l_posterior
        else:
            break

    return i, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior


# Carregando o arquivo CSV e tratando
dados = pd.read_csv('breast_cancer.csv', usecols=['Clump Thickness', 'Class'])
dados = dados.sort_values(by='Clump Thickness')
dados['Class'] = dados['Class'].map({2: 0, 4: 1})  #Troca valores da coluna class para 0 e 1
dados = dados.drop_duplicates()

datasetx = dados['Clump Thickness'].tolist()
datasetx = normalizar(datasetx) #normalizando x
datasety = dados['Class'].tolist()


iteracoes, a_final, b_final, c_final, d_final, k_final, l_final = gradiente_descendente(0.1, 0.1, 2, 5, 8, 0.1, 10**(-4), 0.1)
acc = accuracy(datasetx, datasety, a_final, b_final, c_final, d_final, k_final, l_final)
f1 = f1_score(datasetx, datasety, a_final, b_final, c_final, d_final, k_final, l_final)

print(f"Número de iterações realizadas: {iteracoes}\n Coeficientes finais\n A: {a_final}\n B: {b_final}\n C: {c_final}\n D: {d_final}\n K: {k_final}\n L: {l_final} \n\n Acurácia: {acc:.2f}\n f1 score:{f1:.2f}")

# Gerar a predição final usando os coeficientes ajustados
predicoes = []
for x in datasetx:
    u = sigmoid(a_final * x + b_final)
    w = sigmoid(c_final * u + d_final)
    z = sigmoid(k_final * w + l_final)
    predicoes.append(1 if z >= 0.5 else 0)

# Plotando os dados reais e a linha da sigmoid
plt.plot(datasetx, datasety, 'bo', label='Pontos do dataset')
plt.plot(datasetx, predicoes, 'r-', label='Linha da sigmoid')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rede neural')
plt.legend()
plt.grid(True)
plt.show()