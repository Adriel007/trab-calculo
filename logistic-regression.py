import matplotlib.pyplot as plt
from PIL import Image
#import tensorflow as tf

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



# Mostrar resultado final do pre-processamento de imagem:

#matrix2img(resize_image(img2gray(img2matrix(get_images(paths['training'], paths['keys']['polar'])[0])), 64, 64), 'exemplo.jpg')
