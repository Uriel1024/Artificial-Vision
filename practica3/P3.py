
import numpy as np
from PIL import Image
import os 


# Suma y cuenta los adyacentes no cero (8 vecinos)
def suma(arr,i,j):
    suma = 0
    cuenta = 0
    filas, columnas = arr.shape
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ni, nj = i + dx, j + dy
            if 0 <= ni < filas and 0 <= nj < columnas:
                if arr[ni, nj] != 0:
                    suma += arr[ni, nj]
                    cuenta += 1
    if cuenta == 0:
        return 0
    return suma / cuenta 

#Crea el arreglo de prediccion
def crear_pre(arr):
    arr_pre = np.zeros_like(arr)
    arr_pre[0, :] = arr[0, :]
    arr_pre[:, 0] = arr[:, 0]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i != 0 and j != 0:
                arr_pre[i, j] = suma(arr, i, j)
    return arr_pre

#Crea la imagen comprimida
def crear_comprimida(arr, bit):
    min_val,max_val = min_max(arr)
    if max_val == min_val:
        return np.zeros_like(arr, dtype=int), min_val, 0
    # Calcula el tamaño de cada intervalo
    intervalo = (max_val - min_val) / (2**bit)
    # Asigna código de intervalo a cada valor
    codigos = ((arr - min_val) / intervalo).astype(int)
    codigos = np.clip(codigos, 0, (2**bit) - 1).astype(int)
    return codigos

# Recupera la imagen usando el valor central de cada intervalo
def crear_recuperada(arr_com,arr_pre,arr_error,bit):
    min_val,max_val = min_max(arr_error)
    if max_val == min_val:
        return arr_pre
    intervalo = (max_val - min_val) / (2**bit)
    arr_recuperada = min_val + (arr_com + 0.5) * intervalo
    arr_recuperada += arr_pre
    arr_recuperada = np.round(arr_recuperada)
    return arr_recuperada

def min_max(arr):
    return np.min(arr), np.max(arr)
 
def sel_img():
    print("\n\n\n1.Imagen de berserk.")
    print("2.Imagen de evangelon.")
    print("3.Imagen de evangelon(variante).")
    print("4.Imagen de Berserk(variante).")
    print("5.Imagen de perfect blue.")
    print("6.Imagen de The bends.")
    print("7.Imagen de Full Metal Alchemist.")
    while True:
        op = input("Selecciona una opcion para poder aplicarle ruido(salt & pepper) y regresarla a su forma original:  ")
        if op in ['1','2','3','4','5','6','7']:
            return Image.open(f'img{op}.jpeg')
            break
    else:
        print("Ingresa una opcion valida")



if __name__ == "__main__":
    print("Bienvenido a la tercer practica de vision artificial   .")
    
    while True:

        img = sel_img().convert('L')
        img.show()
        

        arr = np.array(img)
        arr_pre = crear_pre(arr)
        arr_error = arr - arr_pre
        bit = int(input("Dame el numero de bits para comprimir:"))
        arr_com = crear_comprimida(arr_error, bit)
        arr_recuperada = crear_recuperada(arr_com, arr_pre, arr_error, bit)

        # Mostrar la imagen comprimida
        img_comprimida = Image.fromarray((arr_com * (255 // (2**bit - 1))).astype(np.uint8))
        img_comprimida.show(title="Imagen Comprimida")

        # Mostrar la imagen recuperada
        img_recuperada = Image.fromarray(arr_recuperada.astype(np.uint8))
        img_recuperada.show(title="Imagen Recuperada")
        img_recuperada.save("recuperada.png")
        op = input("Deseas salir del programa presiona -1:").lower()
        if op in ["exit",'salir','fuera','acabar', '-1', 'terminar']:
            break
    print("Programa finalizado.")
