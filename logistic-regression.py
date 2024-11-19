import matplotlib.pyplot as plt
import math as math
import numpy as np
import random
from PIL import Image

debugger = True

if debugger:
    file = open('debug.txt', 'w')
    file.write("")
    file.close()

def debug(localsVar):
    if debugger:
        file = open('debug.txt', 'a')
        text = str(localsVar).split(", ")
        oldText = open('debug.txt', 'r').read()
        for i in range(len(text)):
            if text[i] not in oldText:
                file.write(text[i] + ", ")
        file.write("\n")
        file.close()

paths = {
    'test': {
        'path': './test/bears/',
        'length': 8
    },
    'training': {
        'path': './training/bears/',
        'length': 60
    },
    'pandas': {
        'path': './training/pandas/',
        'length': 3
    },
    'keys': {
        'polar': 'polar_bear',
        'black': 'black',
        'panda': 'panda'
    }
}

def plot_points(datasetx, datasety, a, b, c, xlabel="X", ylabel="Y", title="Graph with Sigmoid"):
    plt.scatter(datasetx, datasety, color='blue', label="Data Points")

    sigmoid_values = [sigmoid(a * x**2 + b * x + c) for x in datasetx]

    plt.plot(datasetx, sigmoid_values, color='red', label="Sigmoid Line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
###################################### TRATAMENTO DE IMAGEM ###############################################

def get_images(path_object, key):
    images = []
    for i in range(1, path_object['length'] + 1):
        images.append(path_object['path'] + key + '/' + key + str(i) + '.jpg')
    return images

def img2gray(matrix_RGB):
    for i in range(len(matrix_RGB)):
        for j in range(len(matrix_RGB[0])):
            # Modifica pixel para escala de cinza por meio de média aritmética
            matrix_RGB[i][j] = (matrix_RGB[i][j][0] + matrix_RGB[i][j][1] + matrix_RGB[i][j][2]) // 3
    return matrix_RGB 

def resize_image(matrix_BW, new_height, new_width):
    old_height = len(matrix_BW)
    old_width = len(matrix_BW[0])

    # Matrix resultado vazia com o tamanho correto
    resized_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    # Razões entre escalas
    row_scale = old_height // new_height
    col_scale = old_width // new_width

    for i in range(new_height):
        for j in range(new_width):
            # Mapeia a posição do pixel novo para o do pixel antigo
            old_i = i * row_scale
            old_j = j * col_scale

            # Interpolação bilinear simples
            top_left_i = int(old_i)
            top_left_j = int(old_j)

            # Verifica limites da imagem original
            if top_left_i >= old_height - 1:
                top_left_i = old_height - 2
            if top_left_j >= old_width - 1:
                top_left_j = old_width - 2

            bottom_right_i = top_left_i + 1
            bottom_right_j = top_left_j + 1

            # Fração para interpolação
            delta_i = old_i - top_left_i
            delta_j = old_j - top_left_j

            # Atribui os 4 vizinhos
            top_left = matrix_BW[top_left_i][top_left_j]
            top_right = matrix_BW[top_left_i][bottom_right_j]
            bottom_left = matrix_BW[bottom_right_i][top_left_j]
            bottom_right = matrix_BW[bottom_right_i][bottom_right_j]

            # Interpolação bilinear
            top = (1 - delta_j) * top_left + delta_j * top_right
            bottom = (1 - delta_j) * bottom_left + delta_j * bottom_right
            value = (1 - delta_i) * top + delta_i * bottom

            # Atribui o valor interpolado ao pixel novo
            resized_image[i][j] = value

    return resized_image

def img2matrix(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    width, height = img.size

    rgb_matrix = []

    for y in range(height):
        row = []
        for x in range(width):
            # Obtém o valor RGB de cada pixel
            r, g, b = img.getpixel((x, y))
            row.append((r, g, b))
        rgb_matrix.append(row)

    return rgb_matrix

def matrix2img(matrix_BW, filename):
    # Dimensões da matriz
    height = len(matrix_BW)
    width = len(matrix_BW[0])

    # Cria uma nova imagem em tons de cinza (mode 'L' é grayscale)
    img = Image.new('L', (width, height))

    # Preenche a imagem com os valores da matriz
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), matrix_BW[y][x])

    img.save(filename)
    return True

def matrix2vec(matrix_BW):
    vector = []
    for i in range(len(matrix_BW)):
        for j in range(len(matrix_BW[0])):
            vector.append(matrix_BW[i][j])
    return vector

######################################### CALCULO ###################################################
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Gradientes
def gradDS2(datasetx, datasety, a, b, c):
    grad_a = 0
    grad_b = 0
    grad_c = 0

    for i in range(len(datasetx)):
        pred = a * datasetx[i]**2 + b * datasetx[i] + c
        grad_a += 2 * (pred - datasety[i]) * (datasetx[i]**2)
        grad_b += 2 * (pred - datasety[i]) * (datasetx[i])
        grad_c += 2 * (pred - datasety[i])

    return (grad_a, grad_b, grad_c)

# Funções de a, b e c
def fa_quad(a, b, c, xn):
    exp = -(a*xn**2 + b*xn + c)
    exp_term = math.exp(exp)
    result = xn**2 * exp_term / (1 + exp_term)**2
    print (exp, exp_term, result)
    return result

def fb_quad(a, b, c, xn):
    exp = -(a*xn**2 + b*xn + c)
    print (exp)
    exp_term = math.exp(exp)
    result = xn * exp_term / (1 + exp_term)**2
    return result

def fc_quad(a, b, c, xn):
    exp = -(a*xn**2 + b*xn + c)
    print (exp)
    exp_term = math.exp(exp)
    result = exp_term / (1 + exp_term)**2
    return result

# Gradientes
def gradDS_logistic_quad(datasetx, datasety, a, b, c):
    grad_a = 0
    grad_b = 0
    grad_c = 0
    for i in range(len(datasetx)):
        pred = a * datasetx[i]**2 + b * datasetx[i] + c
        grad_a += 2 * (datasety[i] - pred) * fa_quad(a, b, c, datasetx[i])
        grad_b += 2 * (datasety[i] - pred) * fb_quad(a, b, c, datasetx[i])
        grad_c += 2 * (datasety[i] - pred) * fc_quad(a, b, c, datasetx[i])
        debug(locals())

    return (grad_a, grad_b, grad_c)

def dist2(x_n, y_n, z_n, x_0, y_0, z_0):
    return ((x_n - x_0)**2 + (y_n - y_0)**2 + (z_n - z_0)**2)**0.5

def gradDS_quad(datasetx, datasety, a_0, b_0, c_0, tol, lr):
    a_n = a_0
    b_n = b_0
    c_n = c_0
    
    a_n1 = 99999999
    b_n1 = 99999999
    c_n1 = 99999999
    
    i = 0

    while True:
        grad_a, grad_b, grad_c = gradDS_logistic_quad(datasetx, datasety, a_n, b_n, c_n)
        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        c_n1 = c_n - lr * grad_c
        i += 1
        
        if dist2(a_n1, b_n1, c_n1, a_n, b_n, c_n) <= tol:
            break
        else:
            a_n = a_n1
            b_n = b_n1
            c_n = c_n1

    return (i, a_n, b_n, c_n)

################################### USE #################################################


########### TRAINING ###########
x_list = []
y_list = []

polar = get_images(paths['training'], paths['keys']['polar'])
black = get_images(paths['training'], paths['keys']['black'])

# 2 FORS PARA DEIXAR ORGANIZADO

for i in range(paths['training']['length']):
    black_matrix = img2matrix(black[i])
    black_matrix = resize_image(img2gray(black_matrix), 64, 64)
    black_matrix = matrix2vec(black_matrix)
    black_simplified = sum(black_matrix) // len(black_matrix) # SIMPLIFICAÇÃO POR MÉDIA ( RECOMENDADO SERIA ALGO COMO PCA )
    x_list.append(black_simplified)
    y_list.append(0)

for i in range(paths['training']['length']):
    polar_matrix = img2matrix(polar[i])
    polar_matrix = resize_image(img2gray(polar_matrix), 64, 64)
    polar_matrix = matrix2vec(polar_matrix)
    polar_simplified = sum(polar_matrix) // len(polar_matrix) # SIMPLIFICAÇÃO POR MÉDIA ( RECOMENDADO SERIA ALGO COMO PCA )
    x_list.append(polar_simplified)
    y_list.append(1)

a_0 = random.uniform(min(x_list), max(x_list)) # Dados aleatorios proximos ao valores do dataset    
b_0 = random.uniform(min(x_list), max(x_list)) # Dados aleatorios proximos ao valores do dataset    
c_0 = random.uniform(min(x_list), max(x_list)) # Dados aleatorios proximos ao valores do dataset    
learning_rate = 0.01
tolerance = 1e-6
    
iterations, a_opt, b_opt, c_opt = gradDS_quad(x_list, y_list, a_0, b_0, c_0, tolerance, learning_rate)
    
print (x_list)

plot_points(x_list, y_list, a_opt, b_opt, c_opt, "X", "Y", "Graph of Points")


# Mostrar resultado final do pre-processamento de imagem:
# matrix2img(resize_image(img2gray(img2matrix(get_images(paths['training'], paths['keys']['polar'])[0])), 64, 64), 'exemplo.jpg')