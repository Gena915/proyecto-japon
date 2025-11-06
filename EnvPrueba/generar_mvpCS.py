# ESTE ES EL PRIMER SCRIPT


# import cv2
# from ultralytics import YOLO
# import time
# import os

# # === CONFIGURACIÓN ===
# MODEL_PATH = "best.pt"
# VIDEO_PATH = "video2.mp4" # Asegúrate de que esta ruta sea correcta
# CONFIDENCE_THRESHOLD = 0.3  # Usamos 0.2, si falla, baja a 0.01

# # === CONTADORES Y DEFINICIONES GLOBALES PARA EL REPORTE ===
# total_frames = 0
# frames_con_error = 0
# frames_con_columna = 0
# detecciones_totales = {
#     'columna_correcta': 0,
#     'error_apilado': 0
# }
# segmento_tiene_error = False
# segmento_tiene_columna = False


# # === CÓDIGO DE DESPLIEGUE ===
# def deploy_opencv_frame(model_path, video_path, conf_threshold):
#     """Carga el modelo y despliega la detección en el video usando OpenCV."""

#     # CARGAR MODELO
#     try:
#         model = YOLO(model_path)
#     except Exception as e:
#         print(f"❌ ERROR: No se pudo cargar el modelo YOLO desde {model_path}. ¿Está la ruta correcta? {e}")
#         return

#     #  CARGAR VIDEO
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ ERROR: No se pudo abrir el archivo de video: {video_path}")
#         return

#     # Ventana de visualización
#     cv2.namedWindow("YOLOv8 Deteccion", cv2.WINDOW_NORMAL)

#     print("🚀 Iniciando despliegue. Presiona 'q' para salir.")
    
#     start_time = time.time()
#     frame_count = 0

#     #  PROCESAR FRAME POR FRAME
#     while cap.isOpened():
#         ret, frame = cap.read() # Leer un frame
#         if not ret:
#             break

#         # Ejecutar la inferencia
#         #  640 como imgsz, tal como entrenamiento
#         results = model.predict(source=frame,
#                                 conf=conf_threshold,
#                                 imgsz=640,
#                                 verbose=True) # verbose=False para limpiar la consola

        

#         # Devuelve el frame con las cajas dibujadas.
#         annotated_frame = results[0].plot()

#         # INSERCIÓN AQUÍ: LÓGICA DE CONTEO Y CLAVE GLOBAL


#         global total_frames, frames_con_error, frames_con_columna, detecciones_totales
        
#         global segmento_tiene_error, segmento_tiene_columna 
        
#         # Obtener las clases detectadas en este frame
#         clases_detectadas = results[0].boxes.cls.tolist()
#         clase_nombres = model.names
        
#         has_error_in_frame = False
#         has_columna_in_frame = False
        
#         for cls_index in clases_detectadas:
#             cls_name = clase_nombres[int(cls_index)]
            
#             #  Conteo Total (se usa en el reporte, aunque sea inflado)
#             detecciones_totales[cls_name] = detecciones_totales.get(cls_name, 0) + 1
            
#             if 'error' in cls_name.lower():
#                 has_error_in_frame = True
#                 #  Conteo Único ( solo saber si  error existió en el segmento)
#                 segmento_tiene_error = True 
#             elif 'columna' in cls_name.lower():
#                 has_columna_in_frame = True
#                 segmento_tiene_columna = True
                
#         # Conteo por Frame (se usa para las métricas de % de frames)
#         if has_error_in_frame:
#             frames_con_error += 1
#         if has_columna_in_frame:
#             frames_con_columna += 1
        
        
#         # DESPLEGAR
#         cv2.imshow("YOLOv8 Deteccion", annotated_frame)
        
#         frame_count += 1

#         # Controlar la salida con la tecla 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     #  LIMPIEZA
#     cap.release()
#     cv2.destroyAllWindows()
    
#     end_time = time.time()
#     tiempo_total = end_time - start_time 
    
#     fps = frame_count / (tiempo_total)
#     print(f"✅ Despliegue finalizado. FPS promedio: {fps:.2f}")

#     # REPORTE DIRECTO EN CONSOLA 
#     total_frames_procesados = frame_count # Usamos frame_count ya que es el contador total

#     # Cálculo de métricas
#     tasa_deteccion_error = (frames_con_error / total_frames_procesados) * 100 if total_frames_procesados > 0 else 0

#     # ... código de limpieza y cálculo de tiempo

#     # AJUSTE EN LA IMPRESIÓN DEL REPORTE FINAL

#     print(" REPORTE DE CONTROL DE CALIDAD (MVP)  ")
#     print("------------------------------------------------")
#     print(" MÉTRICAS DE CALIDAD POR SEGMENTO  ")
#     print("------------------------------------------------")
#     #  conteo binario (True/False)
#     print(f"Estado de Columna Detectada: {'✅ SI' if segmento_tiene_columna else '❌ NO'} ")
#     print(f"Estado de Error Detectado: {'❌ ERROR' if segmento_tiene_error else '✅ OK'} ")
#     print("------------------------------------------------")
#     # Mantenemos las métricas infladas como referencia
#     print(f"Frames con Error Detectado: {frames_con_error} (Tasa Inflada: {tasa_deteccion_error:.2f}%)")
#     print("================================================")

# if __name__ == "__main__":
#     deploy_opencv_frame(MODEL_PATH, VIDEO_PATH, CONFIDENCE_THRESHOLD)



# ULTIMO SCRIPT NUEVO

# import cv2
# from ultralytics import YOLO
# import time
# import os
# # IMPORTAR AQUÍ LA LIBRERÍA DE COMUNICACIÓN PLC (Ej: import pycomm3)

# # ==========================================================
# # === 0. CONFIGURACIÓN GENERAL Y RUTAS ===
# # ==========================================================
# MODEL_SUP_PATH = "modelos/best.pt"  # Asumiendo que el modelo final se llama best.pt
# MODEL_LAT_PATH = "modelos/modelo_lateral_anomalias.pt"
# CAM_SUP_ID = 0 
# CAM_LAT_ID = 1 
# CONFIDENCE_THRESHOLD = 0.5 

# # --- CLASES DE VISIÓN ---
# CLASES_ANOMALIA_LATERAL = ['envase_caido_posicion', 'fila_invertida_orientacion', 'stack_desalineado'] 
# CLASE_BORDE_Z = 'borde_z_envase'
# CLASE_REF_SUJECION = 'referencia_sujecion' 
# CLASES_FALLO_SUPERIOR = ['error_apilado', 'segmento_vacio', 'error_producto', 'desalineacion_y_abanico']
# CLASE_POSICION = 'posicion_columna' 
# CLASE_VACIO = 'posicion_vacia' 
# TOTAL_POSICIONES = 8 

# # --- CÓDIGOS Y REGISTROS PLC ---
# CODIGO_PETICION_VISION = 99
# CODIGO_RESPUESTA_OK = 0
# CODIGO_RESPUESTA_FALLO_QC = 1 
# CODIGO_RESPUESTA_PARADA = 2 

# REGISTRO_RESPUESTA = "DB_VIS.Respuesta"
# REGISTRO_CONTEO = "DB_VIS.FilasRestantes"
# REGISTRO_CORRECCION_Y = "DB_VIS.CorreccionY"
# REGISTRO_CORRECCION_Z = "DB_VIS.CorreccionZ" 

# # --- CALIBRACIÓN Z (DEBEN SER AJUSTADOS) ---
# FACTOR_ESCALA_MM_PIXEL = 0.050 # Milímetros por píxel (ejemplo)
# TOLERANCIA_Z_CMM = 5          # Compensar solo si el error es mayor a 0.05 mm (5 cMM)

# # --- CALIBRACIÓN CORRECCIÓN Y (Brazo 1 - Abanico) ---
# # 🚨 AJUSTAR ESTOS VALORES 🚨
# ANCHO_NORMAL_PX = 100            # Ancho promedio de la caja de una columna alineada.
# FACTOR_CALIBRACION_PLC_Y = 0.05  # Factor: (Unidades PLC de Empuje) / (Píxeles de Error).
# UMBRAL_MINIMO_ERROR_PX = 10      # Error mínimo de ancho para activar la compensación.
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # === 1. FUNCIONES SIMULADAS (REEMPLAZAR POR LÓGICA PLC REAL) ===
# # ==========================================================

# def simular_lectura_plc(registro):
#     """Simula la lectura del registro del PLC."""
#     # SIMULACIÓN DE LECTURA (MANTENIDA POR EL SCOPE LOCAL)
#     if not hasattr(simular_lectura_plc, 'ciclo_activo'):
#         simular_lectura_plc.ciclo_activo = True
#         return CODIGO_PETICION_VISION
#     if simular_lectura_plc.ciclo_activo:
#         time.sleep(1)
#         simular_lectura_plc.ciclo_activo = False
#         return 0
#     return 0

# def simular_escritura_plc(registro, valor):
#     """Simula la escritura de la respuesta al PLC."""
#     # En una implementación real, aquí se usaría la librería PLC (ej. pycomm3)
#     print(f"🤖 PLC WRITE: Registro {registro} <- Valor {valor}")
#     time.sleep(0.01)
#     pass


# # ==========================================================
# # === 2. FUNCIONES DE HARDWARE Y UTILIDAD ===
# # ==========================================================

# def cargar_modelos(path_sup, path_lat):
#     try:
#         model_sup = YOLO(path_sup)
#         model_lat = YOLO(path_lat)
#         return model_sup, model_lat
#     except Exception as e:
#         print(f"❌ ERROR al cargar modelos: {e}")
#         return None, None

# def inicializar_camaras(id_sup, id_lat):
#     cap_sup = cv2.VideoCapture(id_sup)
#     cap_lat = cv2.VideoCapture(id_lat)
#     if not cap_sup.isOpened() or not cap_lat.isOpened():
#         print("❌ ERROR: No se puede abrir una o ambas cámaras.")
#         return None, None
#     return cap_sup, cap_lat

# def tomar_frame(cap):
#     ret, frame = cap.read()
#     return frame if ret else None


# # ==========================================================
# # === 3. FUNCIONES DE CÁLCULO DE CORRECCIÓN ===
# # ==========================================================

# def calcular_correccion_y(model_sup, results_boxes):
#     """Calcula el valor de corrección Y (abananamiento) basado en la detección."""
    
#     correccion_y_unidades_plc = 0
#     requiere_correccion = False
    
#     # Usamos las constantes globales definidas al inicio del script
#     W_NORMAL = ANCHO_NORMAL_PX
#     FACTOR_PLC = FACTOR_CALIBRACION_PLC_Y
#     UMBRAL_ERROR = UMBRAL_MINIMO_ERROR_PX
    
#     for box in results_boxes:
#         cls_name = model_sup.names.get(int(box.cls.item())) 
        
#         if cls_name == 'desalineacion_y_abanico':
#             # Coordenadas X:
#             xmin = box.xyxy[0][0].item()
#             xmax = box.xyxy[0][2].item()
#             ancho_actual_px = xmax - xmin
            
#             # Cálculo del error:
#             W_error = ancho_actual_px - W_NORMAL
            
#             # 🚨 CONDICIÓN DE ACTIVACIÓN:
#             if W_error > UMBRAL_ERROR:
#                 requiere_correccion = True
                
#                 # Cálculo de la magnitud de la compensación:
#                 MAGNITUD_COMPENSACION_PLC = W_error * FACTOR_PLC
                
#                 # El valor a enviar es la magnitud del empuje
#                 correccion_y_unidades_plc = abs(MAGNITUD_COMPENSACION_PLC)
                
#                 # Ya que entrenamos el modelo para detectar solo la columna de agarre, 
#                 # salimos del bucle al encontrar el primer error que requiere corrección.
#                 break 
            
#     # Devuelve el valor de corrección en unidades PLC (0 si no hay error significativo)
#     return correccion_y_unidades_plc, requiere_correccion


# def calcular_correccion_z(model_lat, results_boxes):
#     """Calcula la diferencia de altura Z (compensación por falta de apriete)."""
    
#     y_envase = None      # Y_min del envase (más arriba en el frame)
#     y_sujecion = None    # Y_min de la referencia fija
    
#     for box in results_boxes:
#         cls_name = model_lat.names.get(int(box.cls.item()))
#         y_min = box.xyxy[0][1].item() 

#         if cls_name == CLASE_BORDE_Z:
#             if y_envase is None or y_min < y_envase:
#                  y_envase = y_min 
        
#         elif cls_name == CLASE_REF_SUJECION:
#             if y_sujecion is None or y_min < y_sujecion:
#                  y_sujecion = y_min

#     if y_envase is None or y_sujecion is None:
#         return 0, False # No se puede calcular

#     # 1. CÁLCULO EN PÍXELES (Delta P)
#     delta_p = abs(y_envase - y_sujecion) 

#     # 2. CONVERSIÓN A CENTÉSIMAS DE MILÍMETRO (cMM)
#     correccion_z_cMM = int(delta_p * FACTOR_ESCALA_MM_PIXEL * 100)

#     # 3. FILTRO POR TOLERANCIA
#     if correccion_z_cMM > TOLERANCIA_Z_CMM: 
#         return correccion_z_cMM, True
#     else:
#         return 0, False


# # ==========================================================
# # === 4. FUNCIONES DE INFERENCIA ===
# # ==========================================================

# def ejecutar_inferencia_lateral(model_lat, frame):
#     """Ejecuta inferencia en la cámara lateral (SEGURIDAD y CORRECCIÓN Z)."""
#     results = model_lat.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
#     annotated_lat = results[0].plot()
    
#     response_code = CODIGO_RESPUESTA_OK
#     has_critical_error = False

#     # A. EVALUACIÓN DE ANOMALÍAS CRÍTICAS (PARADA)
#     for box in results[0].boxes:
#         cls_name = model_lat.names.get(int(box.cls.item()))
#         if cls_name in CLASES_ANOMALIA_LATERAL:
#             print(f"🚨 Anomalía Lateral Crítica: {cls_name} detectada.")
#             response_code = CODIGO_RESPUESTA_PARADA
#             has_critical_error = True
#             break
            
#     # B. CÁLCULO DE CORRECCIÓN Z
#     if has_critical_error:
#         correccion_z = 0 
#     else:
#         correccion_z, _ = calcular_correccion_z(model_lat, results[0].boxes)
        
#     return response_code, annotated_lat, correccion_z

# def ejecutar_inferencia_superior(model_sup, frame):
#     """
#     Ejecuta la detección superior: 
#     1. Evalúa Fallo de Calidad (QC) y Corrección Y.
#     2. Cuenta las filas restantes.
#     """
#     results = model_sup.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
#     annotated_sup = results[0].plot()







import cv2
from ultralytics import YOLO
import time
import os

# ==========================================================
# === 0. CONFIGURACIÓN Y RUTAS CLAVE ===
# ==========================================================
# 🚨 RUTA CORREGIDA: Debe coincidir con la salida de tu entrenamiento
MODEL_PATH = "bestCS.pt" 
VIDEO_PATH = "videoCS.mp4" 
CONFIDENCE_THRESHOLD = 0.3

# === CONTADORES Y DEFINICIONES GLOBALES PARA EL REPORTE ===
total_frames = 0
frames_con_error = 0
frames_con_columna = 0
frames_con_vacio = 0
frames_con_abanico = 0

# Detecciones totales por clase (para referencia)
detecciones_totales = {} 

# Estado binario por segmento (Si al menos una vez ocurrió en el video)
segmento_tiene_error = False
segmento_tiene_columna = False
segmento_tiene_vacio = False
segmento_tiene_abanico = False


# === 1. FUNCIÓN DE DESPLIEGUE ===
def deploy_opencv_frame(model_path, video_path, conf_threshold):
    """Carga el modelo y despliega la detección en el video usando OpenCV."""

    global total_frames, frames_con_error, frames_con_columna, frames_con_vacio, frames_con_abanico
    global detecciones_totales, segmento_tiene_error, segmento_tiene_columna, segmento_tiene_vacio, segmento_tiene_abanico

    # CARGAR MODELO
    try:
        model = YOLO(model_path)
        clase_nombres = model.names
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar el modelo YOLO desde {model_path}. Revisa la ruta. {e}")
        return

    #  CARGAR VIDEO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo abrir el archivo de video: {video_path}")
        return

    cv2.namedWindow("YOLOv8 Deteccion", cv2.WINDOW_NORMAL)
    print("🚀 Iniciando despliegue. Presiona 'q' para salir.")
    
    start_time = time.time()
    frame_count = 0

    #  PROCESAR FRAME POR FRAME
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1 # Contador global de frames

        # Ejecutar la inferencia (usando imgsz=640 como en entrenamiento)
        results = model.predict(source=frame,
                                conf=conf_threshold,
                                imgsz=640,
                                verbose=False) 

        annotated_frame = results[0].plot()

        # LÓGICA DE CONTEO Y CLAVE GLOBAL
        clases_detectadas = results[0].boxes.cls.tolist()
    
        has_error_in_frame = False

        for cls_index in clases_detectadas:
            cls_name = clase_nombres[int(cls_index)]
    
            # 1. Conteo Total por Clase
            detecciones_totales[cls_name] = detecciones_totales.get(cls_name, 0) + 1
    
            # 2. CONTEO DE ESTADO POR SEGMENTO Y FRAME
            if cls_name == 'error_apilado':
                has_error_in_frame = True
                segmento_tiene_error = True
    
            elif cls_name == 'desalineacion_y_abanico':
                has_error_in_frame = True # El abanico es un fallo/corrección
                frames_con_abanico += 1
                segmento_tiene_abanico = True

            elif cls_name == 'posicion_columna':
                frames_con_columna += 1
                segmento_tiene_columna = True
    
            elif cls_name == 'posicion_vacia':
                frames_con_vacio += 1
                segmento_tiene_vacio = True
    
        # Conteo de Frames con ALGÚN TIPO de Error
        if has_error_in_frame:
            frames_con_error += 1
    
        # DESPLEGAR
        cv2.imshow("YOLOv8 Deteccion", annotated_frame)
    
        frame_count += 1

        # Controlar la salida con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #  LIMPIEZA
    cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    tiempo_total = end_time - start_time 

    # REPORTE FINAL
    total_frames_procesados = frame_count 
    fps = frame_count / (tiempo_total) if tiempo_total > 0 else 0

    print("\n================================================")
    print("📊 REPORTE DE INFERENCIA DEL MODELO SUPERIOR 📊")
    print("================================================")
    print(f"🖼️ Frames Procesados: {total_frames_procesados}")
    print(f"⏱️ Tiempo Total: {tiempo_total:.2f} s")
    print(f"⚡ FPS Promedio: {fps:.2f}")
    print("------------------------------------------------")
    print("🔍 Detección por Estado (Ocurrencia en el video)")
    print("------------------------------------------------")
    # QC y Corrección
    print(f"Fallo QC (Error Apilado): {'❌ FALLO' if segmento_tiene_error else '✅ OK'}")
    print(f"Corrección Y (Abanico): {'📐 REQUERIDA' if segmento_tiene_abanico else '✅ NO'}")
    print(f"Error Total (QC o Y): {'❌ FALLO' if segmento_tiene_error or segmento_tiene_abanico else '✅ OK'}")
    print("------------------------------------------------")
    # Conteo
    print(f"Posición Columna Detectada: {'✅ SI' if segmento_tiene_columna else '❌ NO'}")
    print(f"Posición Vacía Detectada: {'✅ SI' if segmento_tiene_vacio else '❌ NO'}")
    print("================================================")
    print("Conteo de Frames con Fallo:")
    print(f"Frames con Error Apilado/Abanico: {frames_con_error} (Tasa: {(frames_con_error / total_frames_procesados) * 100 if total_frames_procesados > 0 else 0:.2f}%)")


if __name__ == "__main__":
    deploy_opencv_frame(MODEL_PATH, VIDEO_PATH, CONFIDENCE_THRESHOLD)
    
    
    
    