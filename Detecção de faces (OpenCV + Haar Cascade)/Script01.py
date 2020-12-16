import cv2
img = cv2.imread('foto2.jpg')  #Imagem a ser utilizada
detec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Modelo a ser utilizado
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Tons de cinza para otimizar o processo
face = detec.detectMultiScale(cinza, 1.3, 3)  #Detectar a face
for (x, y, larg, alt) in face:  #Desenhar o retângulo
    ret = cv2.rectangle(img, (x, y), (x + larg, y + alt), (0, 255, 0), 3)
cv2.imshow('Detecção de faces', img)  #Mostrar a face detectada
cv2.waitKey(0)