from PIL import Image
import numpy as np


#funcion para aplicar el color, verde, azul rojo a la imagen (manera vertical)
def color_vertical(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_width = width //3


	#para el color rojo
	region = img_filtrada[0: height, 0 : div_width]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,1] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0:height, 0:div_width] = region.astype(np.uint8)


	#para el color verde

	region = img_filtrada[0: height, div_width : div_width * 2 ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0:height, div_width:div_width * 2] = region.astype(np.uint8)


	#para el color azul
	region = img_filtrada[0: height, div_width * 2 : width ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,1] = (1- intensidad)

	img_filtrada[0:height, div_width  * 2 : width] = region.astype(np.uint8)



	return Image.fromarray(img_filtrada)


#funcion para aplicar el color (manera horizontal)
def color_horizontal(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_height = height //3

	#para el color rojo

	region = img_filtrada[0: div_height, 0 : width]
	region = region.astype(np.float32)

	region = np.clip(region,0,255)

	region[...,1] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0: div_height, 0 : width] = region.astype(np.uint8)


	#para el color verde

	region = img_filtrada[div_height : div_height * 2, 0 : width ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[div_height: div_height*2, 0 : width] = region.astype(np.uint8)


	#para el color azul

	region = img_filtrada[div_height*2  : height, 0 : width ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,1] = (1- intensidad)

	img_filtrada[div_height * 2: height, 0 : width] = region.astype(np.uint8)


	return Image.fromarray(img_filtrada)

def pintar_h(imagen, height, width):
	intensidad = .7
	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()

	region = img_filtrada[0: div_height, 0 : width]
	region = region.astype(np.float32)


	#funcion para aplicar el color, verde, azul rojo a la imagen (manera vertical)
def dibujar_H(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_width = width //5

	div_height = height //11
	
	
	#para la primer parte de la imagen(rojo)

	region = img_filtrada[0: height, 0 : div_width  ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,1] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0:height, 0:div_width ] = region.astype(np.uint8)



	#para la primer parte de la imagen(rojo)

	region = img_filtrada[0: height, width - div_width   : width  ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,1] = (1- intensidad)
	region[...,0] = (1- intensidad)

	img_filtrada[0:height, width - div_width  : width] = region.astype(np.uint8)


	#para el segundo | con color verde

	region = img_filtrada[0: height, div_width : div_width * 2 ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0:height, div_width:div_width * 2] = region.astype(np.uint8)


	#para el segundo | con color verde

	region = img_filtrada[0: height, div_width + (2 * div_width ): width - div_width    ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0:height,div_width + (2 * div_width ) : width - div_width ] = region.astype(np.uint8)


	#para el - con color verde

	region = img_filtrada[div_height * 4:div_height *7, div_width * 2: div_width + (2 * div_width ) ]
	region = region.astype(np.float32)


	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[div_height * 4:div_height * 7,div_width * 2: div_width + (2 * div_width ) ] = region.astype(np.uint8)


	return Image.fromarray(img_filtrada)



def menu_imagenes():
	print("\n\n\n1.Imagen de Berserk")
	print("2.Imagen de evangelon")
	print("3.Imagen de evangelon(variante)")
	print("4.Imagen de Berserk")
	print("5.Imagen de perfect blue")
	return input("Selecciona una opcion para poder colorearla(-1 para poder salir): ")




def menu():
	print("\n\n\n\n1.Colorear la imagen de manera vertical.")
	print("2.Colorear la imagen de manera horizontal.")
	print("3.Colorear la inicial de los miembros del equipo (H,U,A).")
	print("-1 Para cambiar de imagen")
	return input("Ingresa una opcion para poder continuar: ")


def seleccion(im, op):
	width, height = im.size
	if op == '1':
		print("\n\n\nColoreando imagen de manera vertical\n\n\n")
		new_im = color_vertical(im , height, width)
		new_im.show()
	elif op =='2':
		print("\n\n\nColoreando imagen de manera horizontal\n\n\n")
		new_im = color_horizontal(im,height,width)
		new_im.show()
	elif op =='3':
		caso = True
		while caso:
			inicial = input("Ingresa la vocal para colorear (H,U,A): ").upper()
			if inicial == 'H':
				print("Se imprimira la inicial de Hector (H)")
				new_im = dibujar_H(im,height,width)
				new_im.show()
				caso = False
			else:
				print(f"{inicial} No es una opcion valida, ingresa una  letra valida(H,U,A).\n\n\n")
				caso = True
	else:
		print("Ingresa una opcon valida (1-3). \n\n\n")

if __name__ == "__main__":
	print("Bienvenido a la primer practica de Vision Artificial.")
	print("Selecciona una imagen para poder continuar.\n\n\n\n")
	sel = ""
	while sel != "-1":
		op1 = ''
		sel = menu_imagenes()
		if sel == '1':
			im = Image.open('img1.jpeg')
			while op1 != '-1':
				op1 = menu()
				if op1 == '-1':
					break
				else:
					seleccion(im,op1)
		elif sel == '2':
			im = Image.open('img2.jpeg')
			while op1 != '-1':
				op1 = menu()
				if op1 == '-1':
					break
				else:
					seleccion(im,op1)
		elif sel == '3':
			im = Image.open('img3.jpeg')
			while op1 != '-1':
				op1 = menu()
				if op1 == '-1':
					break
				else:
					seleccion(im,op1)
		elif sel == '4':
			im = Image.open('img4.jpg')
			while op1 != '-1':
				op1 = menu()
				if op1 == '-1':
					break
				else:
					seleccion(im,op1)
		elif sel == '5':
			im = Image.open('img5.jpeg')
			while op1 != '-1':
				op1 = menu()
				if op1 == '-1':
					break
				else:
					seleccion(im,op1)
		else:
			print("Ingresa una opcion valida para poder continuar (1-5) \n\n\n")
