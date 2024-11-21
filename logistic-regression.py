import matplotlib.pyplot as plt
import math as math
import time
import sys


def load_breast_cancer_data(file_path):
    features = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split(',')
            # Pula o cabeçalho
            try:
                # Pegar a primeira característica 'Clump Thickness'
                features.append(float(values[0]))
                # Transforme a classe: 2 -> 0 (benigno), 4 -> 1 (maligno)
                if int(values[9]) == 2:
                    labels.append(0)
                else:
                    labels.append(1)
            except:
                True
    
    return sort_dataset(features, labels)

def normalizer(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def split_data(features, labels, test_percentage):
    split_index = int(len(features) * (1 - test_percentage))
    train_x = features[:split_index]
    train_y = labels[:split_index]
    test_x = features[split_index:]
    test_y = labels[split_index:]
    return train_x, train_y, test_x, test_y

def getDataset(file_path, test_perc):
    features, labels = load_breast_cancer_data(file_path)
    features = normalizer(features)

    train_x, train_y, test_x, test_y = split_data(features, labels, test_perc)
    return train_x, train_y, test_x, test_y

def sort_dataset(datasetx, datasety):
    # Combinar e ordenar pelo index 0
    data_combined = list(zip(datasetx, datasety))
    data_sorted = sorted(data_combined, key=lambda x: x[0])

    # Separar
    datasetx_sorted, datasety_sorted = zip(*data_sorted)

    return list(datasetx_sorted), list(datasety_sorted)

def plot_points(datasetx, datasety, tsx, tsy, a, b, xlabel="X", ylabel="Y", title="Graph with Sigmoid"):
    plt.scatter(datasetx, datasety, color='blue', label="Training Data")

    sigmoid_values = [sigmoid(a * x + b) for x in datasetx]

    plt.plot(datasetx, sigmoid_values, color='red', label="Sigmoid Line")

    plt.scatter(tsx, tsy, color='green', label="Predictions Data")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

######################################### CALCULO ###################################################
def grad_a(a, b, x, y):
    y_pred = sigmoid(a * x + b)
    return (y_pred - y) * x

def grad_b(a, b, x, y):
    y_pred = sigmoid(a * x + b)
    return (y_pred - y)


def sigmoid(x):
  return 1/(1+math.exp(-x)) 


def gradDS(datasetx, datasety, a, b):
    grad_a_sum = 0
    grad_b_sum = 0

    for i in range(len(datasetx)):
        grad_a_sum += grad_a(a, b, datasetx[i], datasety[i])
        grad_b_sum += grad_b(a, b, datasetx[i], datasety[i])

    return grad_a_sum, grad_b_sum

def dist2(a_0, b_0, a_n, b_n):
    return ((a_n - a_0)**2 + (b_n - b_0)**2)**0.5

def gradienteDescendente(X, Y, a_0, b_0, tol, lr):
    a_n, b_n = a_0, b_0
    a_n1, b_n1 = [99999999] * 2
    i = 0
    
    while True:
        grad_a, grad_b = gradDS(X, Y, a_n, b_n)
        
        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        
        i += 1
        
        err = dist2(a_n, b_n, a_n1, b_n1)
        
        if err > tol:
            a_n, b_n = a_n1, b_n1
        else:
            break
    
    return i, a_n, b_n

################################## MÉTRICAS #############################################

# Função para calcular a Acurácia
def accuracy(dtx, dty, a, b):
    acertos = 0
    for i in range(len(dtx)):
        x = dtx[i]
        y_real = dty[i]
        z = a * x + b
        y_pred = 1 if sigmoid(z) >= 0.5 else 0  # Previsão binária (limiar 0.5)
        if y_pred == y_real:
            acertos += 1
    return acertos / len(dtx)

# Função para calcular o F1 Score
def f1_score(dtx, dty, a, b):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for i in range(len(dtx)):
        x = dtx[i]
        y_real = dty[i]
        z = a * x + b
        y_pred = 1 if sigmoid(z) >= 0.5 else 0  # Previsão binária (limiar 0.5)
        
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

################################### USE #################################################


########### TRAINING ###########
train_x, train_y, test_x, test_y = getDataset("./breast_cancer.csv", 0.1)

a_0 = 0.1
b_0 = 0.1
learning_rate = float(sys.argv[1])
tolerance = float(sys.argv[2])

print(f"Learning Rate: {learning_rate} | Tolerance: {tolerance}")

start = time.time()

iterations, a_opt, b_opt = gradienteDescendente(train_x, train_y, a_0, b_0, tolerance, learning_rate)

acc = accuracy(test_x, test_y, a_opt, b_opt)
f1 = f1_score(test_x, test_y, a_opt, b_opt)

end = time.time()
print(f"Tempo de execução: {end - start}")

print(f"Iterações: {iterations} | a: {a_opt} | b: {b_opt}")
print(f"F1 Score: {f1} | Accuracy: {acc}")

plot_points(train_x, train_y, test_x, test_y, a_opt, b_opt, "X", "Y", "Graph of Points")