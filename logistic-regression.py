import matplotlib.pyplot as plt
import math as math
import time
import sys

def plot_points(datasetx, datasety, tsx, tsy, a, b, c, xlabel="X", ylabel="Y", title="Graph with Sigmoid"):
    plt.scatter(datasetx, datasety, color='blue', label="Training Data")

    sigmoid_values = [sigmoid(a * x**2 + b * x + c) for x in datasetx]

    plt.plot(datasetx, sigmoid_values, color='red', label="Sigmoid Line")

    plt.scatter(tsx, tsy, color='green', label="Predictions Data")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

######################################### CALCULO ###################################################
def grad_a(a, b, c, x, y):
    z = a * x**2 + b * x + c
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**2

def grad_b(a, b, c, x, y):
    z = a * x**2 + b * x + c
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x

def grad_c(a, b, c, x, y):
    z = a * x**2 + b * x + c
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z)

def parabolica(a, b, c, xn, potencia):
    z = a * xn**2 + b * xn + c
    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z) * xn**potencia

def sigmoid(x):
  return 1/(1+math.exp(-x)) 


def gradDS(datasetx, datasety, a, b, c):
    grad_a_sum = 0
    grad_b_sum = 0
    grad_c_sum = 0

    for i in range(len(datasetx)):
        grad_a_sum += grad_a(a, b, c, datasetx[i], datasety[i])
        grad_b_sum += grad_b(a, b, c, datasetx[i], datasety[i])
        grad_c_sum += grad_c(a, b, c, datasetx[i], datasety[i])

    return grad_a_sum, grad_b_sum, grad_c_sum

def dist2_parabolica(a_0, b_0, c_0, a_n, b_n, c_n):
    return ((a_n - a_0)**2 + (b_n - b_0)**2 + (c_n - c_0)**2)**0.5

def gradienteDescendenteParabolica(X, Y, a_0, b_0, c_0, tol, lr):
    # Coeficientes iniciais
    a_n, b_n, c_n = a_0, b_0, c_0
    
    # Próximos coeficientes
    a_n1, b_n1, c_n1 = [99999999] * 3
    
    # Contador de iterações
    i = 0
    
    while True:
        # Gradientes
        grad_a, grad_b, grad_c = gradDS(X, Y, a_n, b_n, c_n)
        
        # Atualizar coeficientes usando gradiente descendente
        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        c_n1 = c_n - lr * grad_c
        
        # Incrementar contador de iterações
        i += 1
        
        # Calcular o erro (distância entre os pontos anteriores e os novos)
        err = dist2_parabolica(a_n, b_n, c_n, a_n1, b_n1, c_n1)
        
        if err > tol:
            # Atualizar os coeficientes para a próxima iteração
            a_n, b_n, c_n = a_n1, b_n1, c_n1
        else:
            # Parar quando o erro for menor que a tolerância
            break
    
    return i, a_n, b_n, c_n

def predictY(x, a, b, c):
    return sigmoid(a * x**2 + b * x + c)

################################## MÉTRICAS #############################################

def precision(y_true, y_pred):
    true_positives = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    false_positives = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

def recall(y_true, y_pred):
    true_positives = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    false_negatives = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

def f1score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def accuracy(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

################################### USE #################################################


########### TRAINING ###########
train_x = []
train_y = []

for i in range(-10, 10, 1):
    train_x.append(i)
    train_y.append(0)

for i in range(20, 40, 1):
    train_x.append(i)
    train_y.append(1)

a_0 = 0.1
b_0 = 0.1
c_0 = 0.1
learning_rate = float(sys.argv[1])
isBashTest = sys.argv[2]
tolerance = 1e-6

print(f"Learning Rate: {learning_rate} | Tolerance: {tolerance}")

start = time.time()

iterations, a_opt, b_opt, c_opt = gradienteDescendenteParabolica(train_x, train_y, a_0, b_0, c_0, tolerance, learning_rate)

########### TESTING ###########
test_x = [0, 3.5, 6.6, 11.5, 17.8, 20]
test_y = []

for x in test_x:
    test_y.append(predictY(x, a_opt, b_opt, c_opt))

f1 = f1score(train_y, test_y)
acc = accuracy(train_y, test_y)

end = time.time()
print(f"Tempo de execução: {end - start}")

print(f"Iterações: {iterations} | a: {a_opt} | b: {b_opt} | c: {c_opt}")
print(f"F1 Score: {f1} | Accuracy: {acc}")

if isBashTest == "False":
    plot_points(train_x, train_y, test_x, test_y, a_opt, b_opt, c_opt, "X", "Y", "Graph of Points")