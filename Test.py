import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gracias a Murtaza's Workshop - Robotics and AI, sin su ayuda este
# archivo quiza no existiria

## ----------------------------- Funciones ----------------------------- #
# Funcion empty para las trackbars
def empty(a):
    pass


# Funcion imshow
def imshow(Img,title = None):
    plt.figure()
    if len(Img.shape) == 3:
        plt.imshow(Img)
    else:
        plt.imshow(Img, cmap='gray')

    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.show() espera a que la imagen se cierre para proseguir


# Funcion imfill (equivalente a imfill de MATLAB)
def imfill(Img):
# Gracias a Satya Mallick por el script
# https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    # Copia la imagen
    Img_floodfill = Img.copy()

    # Mascara usada para el llenado por difusion
    # El tamaño debe ser dos pixeles mayor al de la imagen original
    Ancho, Alto = Img.shape[:2]
    mask = np.zeros((Ancho+2, Alto+2), np.uint8)

    # Llena por difusion (floodfill) desde el punto (0, 0)
    cv2.floodFill(Img_floodfill, mask, (0,0), 255)

    # Imagen llenada invertida
    Img_floodfill_inv = cv2.bitwise_not(Img_floodfill)

    # La imagen final es la union entre la imagen original y la
    # imagen llenada invertida
    Salida = Img | Img_floodfill_inv
    return Salida


# Funcion get_contours
def get_contours(Img,Img_canny):
    Contours, Hierarchy = cv2.findContours(Img_canny,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    # Se obtienen todos los puntos del contorno extremo
    
    Img_contours = Img.copy()
    i = 1
    for cnt in Contours:
        cv2.drawContours(Img_contours,cnt,-1,(0,0,0),3)
        per = cv2.arcLength(cnt,True)
        # Obtencion de puntos de las esquinas
        approx = cv2.approxPolyDP(cnt,0.05*per,True)
        N_esquinas = len(approx)  # Numero de esquinas
        x, y, An, Al = cv2.boundingRect(approx)
        cv2.rectangle(Img_contours,(x,y),(x+An,y+Al),(255,0,0),3)
        
        # Identificacion de formas mediante numero de esquinas
        if N_esquinas == 3: Tipo = "Tri"
        elif N_esquinas == 4: Tipo = "Cuad"
        elif N_esquinas > 4: Tipo = "Circ"
        else: Tipo = "Nada"
        cv2.putText(Img_contours,Tipo,( x+(An//2)-10, y+(Al//2)-10 ),
            cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,0),2)

        i = i + 1

    return Img_contours


# Funcion get_corners
def get_corners(Img,th_1,th_2):
    Canny = cv2.Canny(Img,th_1,th_2)
    Ancho,Alto,Canal = Img.shape
    Contours, Hierarchy = cv2.findContours(Canny,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    # Con esta funcion se obtienen los contornos a partir de los
    # bordes obtenidos por Canny

    for cnt in Contours:
        area = cv2.contourArea(cnt)
        # Calcula el area de cada contorno
        if area > 0.6*Ancho*Alto:
            # Si el area elegida abarca un 60% de la imagen
            # entonces se obtienen las esquinas
            per = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*per,True)
            Esquinas = len(approx)
            # Si el contorno tiene 4 esquinas
            if Esquinas == 4:
                # La funcion devuelve los puntos
                return approx

    # Si no se encuentra un contorno valido devuelve 0
    return 0


# Funcion in_range
def in_range(Img,Min,Max):
    Mask = np.zeros(Img_HSV.shape,np.uint8)
    # Crea una imagen uint8 en negro de dimensiones iguales a Img

    # Mascara Hue
    Mask_min = np.uint8(Img_HSV[:,:,0] >= Min[0] )
    Mask_max = np.uint8(Img_HSV[:,:,0] <= Max[0] )
    # El Hue es la primera componente en los arreglos Min y Max

    # Se agrega una sentencia especial para poder detectar el rojo
    if Min[0] <= Max[0]:
        # Si el valor minimo es menor al maximo se considera como valida
        # la region entre el minimo y el maximo
        Mask[:,:,0] = cv2.bitwise_and( Mask_min, Mask_max )
    else:
        # Si el valor minimo es menor o igual al maximo se considera 
        # valida la region cuyo Hue sea mayor al minimo y menor al maximo
        Mask[:,:,0] = cv2.bitwise_or( Mask_min, Mask_max )

    # Mascara Saturacion
    Mask_min = np.uint8(Img_HSV[:,:,1] >= Min[1] )
    Mask_max = np.uint8(Img_HSV[:,:,1] <= Max[1] )
    # La saturacion es la segunda componente en los arreglos Min y Max

    Mask[:,:,1] = cv2.bitwise_and( Mask_min, Mask_max )
    # cv2.bitwise toma dos argumentos por lo que es necesario unir las
    # mascaras
    Mask[:,:,0] = cv2.bitwise_and( Mask[:,:,0], Mask[:,:,1] )
    
    # Mascara Valor
    Mask_min = np.uint8(Img_HSV[:,:,2] >= Min[2] )
    Mask_max = np.uint8(Img_HSV[:,:,2] <= Max[2] )
    # El valor es la tercera componente en los arreglos Min y Max

    Mask[:,:,1] = cv2.bitwise_and( Mask_min, Mask_max )

    # Como el tipo de dato es uint8, el valor maximo de la mascara debe
    # ser de 255, no 1
    Mask = 255*cv2.bitwise_and( Mask[:,:,0], Mask[:,:,1] )

    # # Para que la mascara pueda ser usada, debe convertirse a uint8
    # Mask = np.uint8(Mask)
    return Mask

## --------------------------------------------------------------------- #



## --------------------- Comandos basicos de Python -------------------- #
# Abrir Imagen a color
Img = cv2.imread('Formas.png')

# La convencion de lectura de cv2.imread (BGR) no es la misma que la de
# plt.imshow (RGB), por lo tanto, se efectua una conversion de color 
# para que se muestre correctamente
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB) # Conversion BGR a RGB

# Mostrar Imagen
imshow(Img, title = 'Imagen Original')
# Convertir la imagen de color a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY) # Conversion
imshow(Img_gris, title = 'Imagen Gris')

# Convertir la imagen gris a blanco y negro mediante umbralado
ret,Img_BW = cv2.threshold(Img_gris,40,255,cv2.THRESH_BINARY)
imshow(Img_BW, title = 'Imagen Blanco y Negro')

# Guardar Imagen
cv2.imwrite("Formas_BW.png",Img_BW)

# Valores Minimos y Maximos
min = np.min(Img)
max = np.max(Img)
print("Maximo: ", max,"\nMinimo: ", min)

# Imrpimir en pantalla el tamaño y el tipo de dato de la imagen
print("Tamaño: ", Img.shape)
print("Tipo: ", Img.dtype)

## ----------------------------- Borrosidad ---------------------------- #
# Abrir Imagen
Img = cv2.imread('Plaqueta.tif')
imshow(Img, title = 'Imagen Original')

# Aplicar borrosidad (Gaussiana)
Blur = cv2.GaussianBlur(Img,(13,13),0)
# El segundo parametro es el kernel, de tamaño 13x13
# El tercer parametro es el sigma, a mayor sigma, mas borrosa la imagen

imshow(Blur, title = 'Imagen Borrosa')

## ------------------------ Deteccion de canny ------------------------- #
Canny = cv2.Canny(Img,100,100,0)
# El segundo y el tercer parametro son los umbrales
# El ultimo parametro es el sigma

imshow(Canny, title = 'Deteccion con Canny')

## ---------------------------- Dilatacion ----------------------------- #
# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((5,5),np.uint8)
Canny_dil = cv2.dilate(Canny,kernel,iterations=1)
# El tercer argumento es opcional y por defecto es 1

imshow(Canny_dil, title = 'Canny + Dilatacion')

## ----------------------------- Erosion ------------------------------- #
Canny_er = cv2.erode(Canny,kernel,iterations=1)
# El tercer argumento es opcional y por defecto es 1

imshow(Canny_er, title = 'Canny + Erosion')

## ----------------------------- Recorte ------------------------------- #
Crop = Img[700:-1,700:-1]
# El -1 es equivalente a la palabra clave 'end' de MATLAB

imshow(Crop, title = 'Imagen Recortada')

## ---------------------- Comandos matriciales RGB --------------------- #
Img = np.zeros((512,512,3),np.uint8)
# Crea una imagen en negro de tipo uint8 y dimension 512x512x3

Img[0:-1,0:170,0]   = 255    # Seteo franja Roja
Img[0:-1,171:342,1] = 255    # Seteo franja Verde
Img[0:-1,343:-1,2]  = 255    # Seteo franja Azul

# Ploteo
imshow(Img, title = 'Prueba de Colores')

## -------------------------- Formas en Opencv ------------------------- #
Img = np.zeros((512,512,3),np.uint8)
# Crea una imagen en negro de tipo uint8 y dimension 512x512x3

cv2.rectangle(Img,(50,50),(462,462),(255,255,255))
# Forma un rectangulo desde (50,50) hasta (462,462) de color blanco

cv2.line(Img,(130,130),(380,380),(255,0,0),30)
# Forma una linea roja de grosor 30 desde el punto (0,0) hasta (512,512)

cv2.circle(Img,(256,256),180,(255,0,0),30)
# Forma un circulo rojo de grosor 30 con centro en (256,256) y radio 50

cv2.putText(Img,"Aguante Pappo viejaaaaaa",(50,30),
            cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),2)
# Texto en verde
cv2.putText(Img,"No me importa nadaaa",(70,500),
            cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),2)
# Texto en azul

imshow(Img, title = 'Prueba de Formas')

## -------------- Transformaciones geometricas con Python -------------- #
Img = cv2.imread('Hoja Respuestas.jpg')
imshow(Img, title = 'Imagen Original')
print(Img.shape)

P_1 = get_corners(Img,100,100)
# A traves de get_corners se obtienen las esquinas automaticamente

Img_esq = Img.copy()
Img_esq = cv2.drawContours(Img_esq,P_1,-1,(0,0,255),25)
imshow(Img_esq, title = 'Esquinas detectadas')

P_1 = np.float32(P_1)
# Se convierte P_1 a float32
Ancho,Alto = 600,800    # Ancho y alto de la nueva imagen
P_2 = np.float32([ [Ancho,0],[0,0],[0,Alto],[Ancho,Alto] ])
# Hay que tener cuidado con el orden de los puntos en P_2, deben
# corresponder a las mismas esquinas que los de la imagen original,
# en el mismo orden

Matriz = cv2.getPerspectiveTransform(P_1,P_2)
# Este comando crea la matriz para girar la imagen
# Ambos conjuntos deben ser de tipo np.float32

Transformada = cv2.warpPerspective(Img,Matriz,(Ancho,Alto))
# Este comando gira la imagen a partir de la matriz obtenida

imshow(Transformada, title = 'Imagen Transformada')

## ---------------- Unir imagenes con comandos de numpy ---------------- #
# Horizontal
Union_H = np.hstack((Transformada,Transformada))
imshow(Union_H, title = 'Union Horizontal')

# Vertical
Union_V = np.vstack((Transformada,Transformada))
imshow(Union_V, title = 'Union Vertical')

## --------------- Trackbars - Diferenciacion de colores --------------- #
# Confeccion de Trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",660,250)

# Para el Hue, Open-CV soporta solo hasta 179
cv2.createTrackbar("Hue Minimo","Trackbars",150,179,empty)
cv2.createTrackbar("Hue Maximo","Trackbars",5,179,empty)

# Para el valor y la saturacion el maximo es 255
cv2.createTrackbar("Sat Minima","Trackbars",142,255,empty)
cv2.createTrackbar("Sat Maxima","Trackbars",255,255,empty)
cv2.createTrackbar("Val Minimo","Trackbars",89,255,empty)
cv2.createTrackbar("Val Maximo","Trackbars",255,255,empty)

while True:
    Img = cv2.imread('Peppers.png')
    # Recordar que cv2.imread lee en formato BGR

    # Por eso se convierte la imagen de BGR a HSV
    Img_HSV = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)

    # Con estos comandos se obtienen valores de los Trackbars
    H_min = cv2.getTrackbarPos("Hue Minimo","Trackbars")
    H_max = cv2.getTrackbarPos("Hue Maximo","Trackbars")
    S_min = cv2.getTrackbarPos("Sat Minima","Trackbars")
    S_max = cv2.getTrackbarPos("Sat Maxima","Trackbars")
    V_min = cv2.getTrackbarPos("Val Minimo","Trackbars")
    V_max = cv2.getTrackbarPos("Val Maximo","Trackbars")
    print(H_min,H_max,S_min,S_max,V_min,V_max)

    Minimo = np.array([H_min, S_min, V_min])
    Maximo = np.array([H_max, S_max, V_max])
    
    # Para crear la mascara se usa una funcion personalizada
    Mask = in_range(Img_HSV, Minimo, Maximo)
    # Esta permite elegir un valor minimo de Hue mayor al maximo
    # Se requiere esto debido a que en HSV el color rojo oscila
    # entre valores mayores a 0 y menores a 180

    cv2.imshow("Imagen", Img)
    cv2.imshow("Mascara", Mask)
    
    # Se busca cerrar el while al presionar 'q'
    if cv2.waitKey(1) == ord('q'):
        # Se aprovecha para ubicar el waitKey en el if
        cv2.destroyAllWindows() # Cierra todas las ventanas
        break


print(Mask.shape,Img_HSV.shape)
print("Tipo Mascara: ", Mask.dtype, "\n" "Tipo Imagen: ", Img.dtype )

Final = cv2.bitwise_and(Img, Img, mask=Mask)
# Para mostrar la imagen mediante plt.imshow se la debe convertir a RGB
Final = cv2.cvtColor(Final,cv2.COLOR_BGR2RGB)

imshow(Final, title = 'Rojo Detectado')

## ------------------------ Deteccion de bordes ------------------------ #
Img = cv2.imread('Formas_2.png')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

Canny = cv2.Canny(Img,100,100)

imshow(Img, title = 'Imagen Original')

Bordes = get_contours(Img,Canny)
imshow(Bordes, title = 'Deteccion de Formas')
# Cabe destacar que la clasificacion de formas mediante el numero de
# esquinas no es muy bueno, puede verse que muestra varios errores

## --------------------------------------------------------------------- #