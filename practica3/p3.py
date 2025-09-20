import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import numpy as np
from PIL import Image


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

# Función para calcular la relación señal/ruido (SNR)
def calcular_snr(original, recuperada):
    original = original.astype(np.float64)
    recuperada = recuperada.astype(np.float64)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - recuperada) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def crear_pre_causal(arr):
    arr_pre = np.zeros_like(arr, dtype=np.float64)
    # El valor predicho en (i, j) es el valor real en (i, j-1)
    arr_pre[:, 1:] = arr[:, :-1]
    arr_pre[0, 0] = arr[0, 0]
    return arr_pre

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Compresión y Recuperación de Imágenes")

        self.img_options = [
            "1. Berserk",
            "2. Evangelion",
            "3. Evangelion (variante)",
            "4. Berserk (variante)",
            "5. Perfect Blue",
            "6. The Bends",
            "7. Full Metal Alchemist"
        ]
        self.img_files = [f"img{i}.jpeg" for i in range(1,8)]

        self.selected_img = tk.StringVar(value=self.img_options[0])
        self.bits = tk.IntVar(value=4)

        # Widgets
        ttk.Label(root, text="Selecciona imagen:").grid(row=0, column=0, sticky="w")
        self.img_menu = ttk.Combobox(root, textvariable=self.selected_img, values=self.img_options, state="readonly", width=30)
        self.img_menu.grid(row=0, column=1, columnspan=2, sticky="w")

        ttk.Label(root, text="Bits de compresión:").grid(row=1, column=0, sticky="w")
        self.bits_entry = ttk.Entry(root, textvariable=self.bits, width=5)
        self.bits_entry.grid(row=1, column=1, sticky="w")

        self.procesar_btn = ttk.Button(root, text="Procesar", command=self.procesar)
        self.procesar_btn.grid(row=1, column=2, sticky="w")

        self.img_labels = []
        self.img_titles = ["Original", "Predicción", "Error"]
        for i, title in enumerate(self.img_titles):
            ttk.Label(root, text=title).grid(row=2, column=i)
            lbl = ttk.Label(root)
            lbl.grid(row=3, column=i, padx=5, pady=5)
            self.img_labels.append(lbl)

        self.img_titles2 = [ "Comprimida", "Recuperada"]
        for i, title in enumerate(self.img_titles2):
            ttk.Label(root, text=title).grid(row=4, column=i)
            lbl = ttk.Label(root)
            lbl.grid(row=5, column=i, padx=5, pady=5)
            self.img_labels.append(lbl)

        self.snr_label = ttk.Label(root, text="SNR: ")
        self.snr_label.grid(row=6, column=0, columnspan=5)

    def procesar(self):
        idx = self.img_options.index(self.selected_img.get())
        img_path = self.img_files[idx]
        bit = self.bits.get()
        if not (1 <= bit <= 8):
            messagebox.showerror("Error", "El número de bits debe estar entre 1 y 8.")
            return
        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen: {e}")
            return
        arr = np.array(img)

        arr_pre = crear_pre_causal(arr)

        arr_pre = crear_pre(arr)
        arr_error = arr - arr_pre
        arr_com = crear_comprimida(arr_error, bit)
        arr_recuperada = crear_recuperada(arr_com, arr_pre, arr_error, bit)

        # Imágenes procesadas
        imgs = [
            img,
            Image.fromarray(np.round(arr_pre).astype(np.uint8)),
            Image.fromarray((np.round(arr_error)+128).astype(np.uint8)),
            Image.fromarray((arr_com * (255 // (2**bit - 1))).astype(np.uint8)),
            Image.fromarray(np.clip(arr_recuperada,0,255).astype(np.uint8))
        ]
        # Nuevo tamaño para las imágenes
        nuevo_tamano = (300, 300)
        for lbl, im in zip(self.img_labels, imgs):
            im_resized = im.resize(nuevo_tamano)
            imtk = ImageTk.PhotoImage(im_resized)
            lbl.imgtk = imtk
            lbl.config(image=imtk)

        snr = calcular_snr(arr, arr_recuperada)
        self.snr_label.config(text=f"SNR: {snr:.2f} dB")

# Ejecutar interfaz
if __name__ == "__main__":
    root = tk.Tk()
    # Aumentar tamaño de la ventana principal
    root.geometry("1600x500")
    app = App(root)
    root.mainloop()

