import cv2
from ultralytics import YOLO
import time
import os

# ==========================================================
# === 1. CONFIGURACIÓN EXCLUSIVA PARA CÁMARA LATERAL (EJE X) ===
# ==========================================================
MODEL_PATH_LATERAL = "modelo_lateral_anomalias.pt" # <<< ¡ACTUALIZAR con la ruta de tu modelo lateral!
VIDEO_PATH_LATERAL = "video_lateral_stream.mp4"    # <<< Usar la ruta del stream o video lateral
CONFIDENCE_THRESHOLD_LATERAL = 0.5  # Umbral de confianza: ajusta si hay muchos Falsos Positivos/Negativos

# 🚨 CLASES QUE DETONAN LA ALARMA/PARADA 🚨
CLASES_ANOMALIA = [
    'envase_caido_posicion',         # Anomalía de posición grave
    'fila_invertida_orientacion',    # Anomalía de orientación
    'stack_desalineado_posicion'     # Anomalía de posición (desplazamiento)
] 
CLASE_PRINCIPAL = 'columna_lateral_ok' # Clase de la columna correcta (para métricas)


# === 2. CONTADORES Y DEFINICIONES GLOBALES PARA EL REPORTE ===
frames_con_anomalia = 0
frames_con_columna_lateral = 0
detecciones_totales_lateral = {}
segmento_tiene_anomalia = False
segmento_tiene_columna_lateral = False


# === 3. FUNCIÓN PRINCIPAL DE DESPLIEGUE ===
def deploy_lateral_anomalias(model_path, video_path, conf_threshold, clases_anomalia):
    """
    Carga el modelo lateral y despliega la detección de anomalías.
    Genera un reporte binario de Anomalía Detectada.
    """
    # CARGAR MODELO
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar el modelo YOLO desde {model_path}. {e}")
        return

    #  CARGAR VIDEO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo abrir el archivo de video: {video_path}")
        return

    cv2.namedWindow("YOLOv8 Deteccion Lateral", cv2.WINDOW_NORMAL)
    print(f"🚀 Iniciando despliegue LATERAL. Modelo: {os.path.basename(model_path)}. Presiona 'q' para salir.")
    
    start_time = time.time()
    frame_count = 0

    #  PROCESAR FRAME POR FRAME
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break

        # Ejecutar la inferencia (verbose=True para monitorear el rendimiento en ms)
        results = model.predict(source=frame,
                                conf=conf_threshold,
                                imgsz=640,
                                verbose=True) # Muestra el rendimiento en consola

        annotated_frame = results[0].plot()

        # === LÓGICA DE CONTEO Y REPORTE BINARIO LATERAL ===
        global frames_con_anomalia, frames_con_columna_lateral, segmento_tiene_anomalia, segmento_tiene_columna_lateral
        
        clase_nombres = model.names
        
        has_anomalia_in_frame = False
        has_columna_in_frame = False
        
        for box in results[0].boxes:
            cls_index = int(box.cls.item())
            cls_name = clase_nombres[cls_index]
            
            #  Conteo Total (inflado)
            detecciones_totales_lateral[cls_name] = detecciones_totales_lateral.get(cls_name, 0) + 1
            
            #  Verificar si la clase es una anomalía crítica
            if cls_name in clases_anomalia:
                has_anomalia_in_frame = True
                segmento_tiene_anomalia = True # Bandera de fallo
            
            #  Verificar si la columna principal está presente
            if cls_name == CLASE_PRINCIPAL:
                has_columna_in_frame = True
                segmento_tiene_columna_lateral = True
                
        # Conteo por Frame (para métricas de recurrencia)
        if has_anomalia_in_frame:
            frames_con_anomalia += 1
        if has_columna_in_frame:
            frames_con_columna_lateral += 1
        
        # DESPLEGAR
        cv2.imshow("YOLOv8 Deteccion Lateral", annotated_frame)
        
        frame_count += 1

        # Controlar la salida con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #  LIMPIEZA Y REPORTE FINAL
    cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    tiempo_total = end_time - start_time 
    
    fps = frame_count / (tiempo_total)
    print(f"\n✅ Despliegue LATERAL finalizado. FPS promedio: {fps:.2f}")

    # CÁLCULOS
    tasa_anomalia_frames = (frames_con_anomalia / frame_count) * 100 if frame_count > 0 else 0
    
    print("\n================================================")
    print("    REPORTE DE ANOMALÍAS CÁMARA LATERAL (EJE X) ")
    print("================================================")
    print(f"🎥 Video Analizado: {VIDEO_PATH_LATERAL}")
    print(f"⏰ Tiempo Total de Análisis: {tiempo_total:.2f} segundos")
    print(f"🖼️ Frames Procesados: {frame_count}")
    print("------------------------------------------------")
    print("             DIAGNÓSTICO BINARIO (Q.C.)         ")
    print("------------------------------------------------")
    # Si segmento_tiene_anomalia es True, el Brazo 1 o la línea debe detenerse/alarmarse.
    print(f"Estado de Columna Detectada: {'✅ SI' if segmento_tiene_columna_lateral else '❌ NO'} ")
    print(f"Anomalía Detectada: {'❌ ANOMALÍA - ¡PARADA!' if segmento_tiene_anomalia else '✅ OK'} ")
    print("------------------------------------------------")
    print("             DETALLE Y RECURRENCIA              ")
    print("------------------------------------------------")
    
    # Muestra la recurrencia de las clases de anomalía
    for cls in clases_anomalia:
        print(f"Total Detecciones ({cls.upper()}): {detecciones_totales_lateral.get(cls, 0)} (Inflado)")

    print(f"Frames con Anomalía: {frames_con_anomalia} (Tasa Inflada: {tasa_anomalia_frames:.2f}%)")
    print("================================================")


if __name__ == "__main__":
    deploy_lateral_anomalias(MODEL_PATH_LATERAL, VIDEO_PATH_LATERAL, CONFIDENCE_THRESHOLD_LATERAL, CLASES_ANOMALIA)















