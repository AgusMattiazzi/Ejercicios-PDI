
from distutils.command.config import config
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Gracias a Murtaza's Workshop - Robotics and AI, sin su ayuda este
# archivo quiza no existiria

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# ## Detección de carateres con openCV
# Img = cv2.imread('1.png')
# Img = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
# Alto,Ancho,Canales = Img.shape

# print(pytesseract.image_to_string(Img))
# B_boxes = pytesseract.image_to_boxes(Img)

# for i in B_boxes.splitlines():
#     print(i)
#     B = i.split(' ')
#     print(B)
#     X_1,Y_1,X_2,Y_2 = int(B[1]), int(B[2]), int(B[3]), int(B[4])
#     cv2.rectangle(Img, (X_1,Alto-Y_1), (X_2,Alto-Y_2), (255,0,0), 2)
#     cv2.putText(Img, i[0], (X_1,Alto-Y_1+20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,50,50), 2)

# cv2.imshow('Resultado', Img)
# cv2.waitKey(0)

# ## Detección de palabras con openCV
# Img = cv2.imread('1.png')
# Img = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
# Alto,Ancho,Canales = Img.shape

# B_boxes = pytesseract.image_to_data(Img)
# print(B_boxes)

# for x,b in enumerate( B_boxes.splitlines() ):
#     ## enumerate permite tener un índice para manejar el for sin generar una variable nueva
#     if x != 0:
#         print(i)
#         b = b.split()
#         print(b)

#         if len(b) == 12:
#             # La información acá se da en otro formato, porque? nadie lo sabe
#             X,Y,An,Al = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(Img, (X,Y), (X+An,Y+Al), (255,0,0), 2)
#             cv2.putText(Img, b[11], (X,Y+Al+25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,50,50), 2)

# cv2.imshow('Resultado', Img)
# cv2.waitKey(0)

## Detección con openCV (sólo dígitos)
Img = cv2.imread('1.png')
Img = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
Alto,Ancho,Canales = Img.shape

conf = r'--oem 3 --psm 6 outputbase digits'
B_boxes = pytesseract.image_to_data(Img, config = conf)
print(B_boxes)

for x,b in enumerate( B_boxes.splitlines() ):
    ## enumerate permite tener un índice para manejar el for sin generar una variable nueva
    if x != 0:
        print(b)
        b = b.split()
        print(b)

        if len(b) == 12:
            # La información acá se da en otro formato, porque? nadie lo sabe
            X,Y,An,Al = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(Img, (X,Y), (X+An,Y+Al), (255,0,0), 2)
            cv2.putText(Img, b[11], (X,Y+Al+25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,50,50), 2)

cv2.imshow('Resultado', Img)
cv2.waitKey(0)