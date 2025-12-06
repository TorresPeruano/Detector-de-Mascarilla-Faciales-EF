import cv2
from ultralytics import YOLO 
import requests
import numpy as np
import os
import random

# --- Configuración del Modelo YOLO ---
# Cargar el modelo YOLOv8 para detección de rostros (nano version)
# El modelo se descargará automáticamente si no existe localmente.
try:
    model = YOLO("yolov8n-face.pt")
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    # Detener el script si el modelo no carga
    exit()

# Enlaces de cualquier sitio web (JPG, PNG, JPEG)
ruta_imagenes = [
    ("Persona1", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS01Zi3y9iIG8HC9_KtCmVHyD95Mw9FVVcONA&s"),
    ("Persona2", "https://www.apeseg.org.pe/wp-content/uploads/2021/06/GettyImages-1215709494-1.jpg"),
    ("Persona3", "https://cloudfront-us-east-1.images.arcpublishing.com/infobae/72B4DFE6ZVHTNGDRRE6IBRMYQ4.jpg"),
]


# Crear carpetas del dataset
os.makedirs("Dataset/Con_Mascarilla", exist_ok=True)
os.makedirs("Dataset/Sin_Mascarilla", exist_ok=True)

def descargar_imagen(url):
    """Descarga una imagen desde una URL y la decodifica usando OpenCV."""
    try:
        response = requests.get(url, timeout=10) # Añadir timeout
        # Verificar si la descarga fue exitosa (código 200)
        if response.status_code != 200:
            print(f"Error HTTP {response.status_code} al descargar la imagen: {url}")
            return None
            
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error de red o timeout al descargar la imagen {url}: {e}")
        return None
    except Exception as e:
        print(f"Error al procesar la imagen {url}: {e}")
        return None

# --- Bucle Principal con Detección YOLOv8 ---
contador = 0

for nombre, url in ruta_imagenes:
    print(f"\n-> Procesando imagen: {nombre} ({url})")
    image = descargar_imagen(url)
    if image is None:
        continue

    # Realizar la inferencia con el modelo YOLO
    # Se recomienda usar 'verbose=False' para no imprimir logs en cada iteración
    results = model(image, verbose=False)

    face_detected = False

    for r in results:
        # 'r.boxes' contiene todos los cuadros delimitadores detectados
        for box in r.boxes:
            # Obtener coordenadas del cuadro (xmin, ymin, xmax, ymax) en píxeles enteros (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # Calcular ancho y alto (formato x, y, w, h)
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            # Asegurar que las coordenadas son válidas
            if w <= 0 or h <= 0:
                continue
            
            # Recortar el rostro
            rostro = image[y:y+h, x:x+w]

            if rostro.size == 0:
                continue

            face_detected = True

            # Clasificación aleatoria simulada
            tiene_mascarilla = random.choice([True, False])
            label = "Con Mascarilla" if tiene_mascarilla else "Sin Mascarilla"
            color = (0, 255, 0) if tiene_mascarilla else (0, 0, 255) # Verde / Rojo

            # Guardar imagen recortada
            file_path = f"Dataset/{label}/{nombre}_{contador:03d}.jpg"
            cv2.imwrite(file_path, rostro)
            contador += 1
            print(f"   [Guardado]: {file_path}")

            # Dibujar cuadros en la imagen original
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    if not face_detected:
        print(f"   [INFO]: No se detectaron rostros en la imagen {nombre}")

    # Mostrar resultado final por cada imagen descargada
    if image is not None:
        cv2.imshow("Detección Facial con YOLOv8", image)
        cv2.waitKey(3000) # Muestra la imagen por 3 segundos

cv2.destroyAllWindows()
print(f"\n Dataset generado correctamente en la carpeta 'Dataset'. Total de rostros guardados: {contador}")