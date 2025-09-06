from PIL import Image
import numpy as np

#funcion para aplicar el color rojo a la imagen (manera vertical)
def color_rojo(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_width = width //3

	region = img_filtrada[0: height, 0 : div_width]
	region = region.astype(np.float32)

	region = np.clip(region,0,255)

	region[...,1] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0: height, 0:div_width] = region.astype(np.uint8)


	return Image.fromarray(img_filtrada)


#funcion para aplicar el color verde a la imagen (manera vertical)
def color_verde(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_width = width //3

	region = img_filtrada[0: height, div_width : div_width  * 2]
	region = region.astype(np.float32)

	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,2] = (1- intensidad)

	img_filtrada[0: height, div_width : div_width * 2] = region.astype(np.uint8)


	return Image.fromarray(img_filtrada)




#funcion para aplicar el color azul a la imagen (manera vertical)
def color_azul(imagen, height, width):
	
	intensidad = .7

	matriz_img = np.array(imagen)

	img_filtrada = matriz_img.copy()


	div_width = width //3

	region = img_filtrada[0: height, div_width * 2 : width]
	region = region.astype(np.float32)

	region = np.clip(region,0,255)

	region[...,0] = (1- intensidad)
	region[...,1] = (1- intensidad)

	img_filtrada[0: height, div_width * 2: width] = region.astype(np.uint8)


	return Image.fromarray(img_filtrada)


if __name__ == "__main__":
	im = Image.open('prueba.jpeg')
	width, height = im.size
	new_im = color_rojo(im , height, width)
	
	new_im = color_verde(new_im, height, width)
	new_im = color_azul(new_im, height, width)
	new_im.show()