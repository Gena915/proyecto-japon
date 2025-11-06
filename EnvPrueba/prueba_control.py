import cv2
from ultralytics import YOLO
import time
import os

# ==========================================================
# === 0. CONFIGURACIÓN GENERAL Y RUTAS (MODO SIMULACIÓN) ===
# ==========================================================
MODEL_SUP_PATH = "bestCS.pt"
MODEL_LAT_PATH = "bestPruebaCL.pt"

# 🚨 ¡PASO CRÍTICO! ACTUALIZA ESTAS RUTAS CON TUS ARCHIVOS DE VIDEO
VIDEO_SUP_PATH = "videoCS.mp4" 
VIDEO_LAT_PATH = "videoCL.mp4" 

# !!! UMBRALES DE CONFIANZA ESPECÍFICOS !!!
CONFIDENCE_SUP = 0.60 # Alto, para reducir ruido y falsas alarmas QC en Superior
CONFIDENCE_LAT = 0.05 # Bajo, para asegurar la detección de etiquetas de medición Z en Lateral

# Clases de Visión
CLASES_ANOMALIA_LATERAL = ['error_caido'] 
CLASES_FALLO_SUPERIOR = ['error_apilado', 'error_alerta']
CLASE_ABANICO_Y = 'error_abanico' # Mantenida por si se necesita para lógica futura

CLASE_POSICION = 'posicion_columna' 
CLASE_VACIO = 'posicion_vacia' 
TOTAL_POSICIONES = 8 

# --- Constantes para Corrección Z ---
CLASE_REFERENCIA = 'referencia_fija' 
CLASE_BORDE_ENV = 'borde_envase'     
CLASE_MITAD_ENV = 'mitad_envase'     

# Valores de Calibración
D_REAL_MM = 100.0 
OFFSET_CERO_PX = 40 
CORRECCION_Y_FIJA_PX = 50 
TOLERANCIA_COLUMNA_PX = 30 # Desviación máxima en X (píxeles) permitida antes de corregir Y.

# NUEVO: Diccionario para almacenar los centros X ideales. Se llenará dinámicamente.
X_CENTROS_IDEALES = {} 

# Códigos de Comunicación PLC (Se mantienen solo como etiquetas para la lógica)
CODIGO_PETICION_VISION = 99
CODIGO_RESPUESTA_OK = 0
CODIGO_RESPUESTA_FALLO_QC = 1 
CODIGO_RESPUESTA_PARADA = 2 

# ==========================================================
# === 1. FUNCIONES DE INICIALIZACIÓN Y CALIBRACIÓN Y ===
# ==========================================================

def cargar_modelos(path_sup, path_lat):
    """Carga los modelos YOLOv8 para ambas cámaras."""
    try:
        model_sup = YOLO(path_sup)
        model_lat = YOLO(path_lat)
        return model_sup, model_lat
    except Exception as e:
        print(f"❌ ERROR al cargar modelos: {e}")
        return None, None

def inicializar_entradas(path_sup, path_lat):
    """Inicializa la captura de video usando rutas de archivo."""
    cap_sup = cv2.VideoCapture(path_sup)
    cap_lat = cv2.VideoCapture(path_lat)
    if not cap_sup.isOpened():
        print(f"❌ ERROR: No se puede abrir el archivo de video Superior: {path_sup}")
        return None, None
    if not cap_lat.isOpened():
        print(f"❌ ERROR: No se puede abrir el archivo de video Lateral: {path_lat}")
        return None, None
    return cap_sup, cap_lat

def calcular_centros_ideales(model_sup, frame_sup):
    """
    Calcula la posición X ideal para cada una de las 8 columnas
    basándose en el promedio de las detecciones reales del primer frame.
    """
    global X_CENTROS_IDEALES, CLASE_POSICION, CLASE_VACIO, TOTAL_POSICIONES
    
    # Usamos un umbral bajo (0.1) para asegurar la captura de todas las columnas de referencia.
    results = model_sup.predict(source=frame_sup, conf=0.1, verbose=False) 
    
    centros_x_detectados = []
    
    for box in results[0].boxes:
        cls_name = model_sup.names.get(int(box.cls.item()))
        if cls_name == CLASE_POSICION or cls_name == CLASE_VACIO:
            x_center = int((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2)
            centros_x_detectados.append(x_center)
            
    if len(centros_x_detectados) < 2:
        print("⚠️ Calibración Y Fallida: Se necesitan al menos 2 columnas para calcular la distancia promedio.")
        X_CENTROS_IDEALES = {} 
        return

    # 1. Calcular la Distancia Promedio (Pitch)
    centros_x_detectados.sort()
    deltas = [centros_x_detectados[i+1] - centros_x_detectados[i] 
            for i in range(len(centros_x_detectados) - 1)]
    
    distancia_ideal_px = sum(deltas) / len(deltas)
    
    # 2. Proyectar las 8 posiciones
    primer_centro_ideal = centros_x_detectados[0] 
    
    X_CENTROS_IDEALES = {}
    for i in range(TOTAL_POSICIONES):
        X_CENTROS_IDEALES[i + 1] = int(primer_centro_ideal + i * distancia_ideal_px)
        
    print(f"✅ Calibración Y Exitosa: Distancia promedio columna: {distancia_ideal_px:.2f} px")
    print(f"   Centros Ideales generados: {X_CENTROS_IDEALES}")

# ==========================================================
# === 2. FUNCIÓN: CÁLCULO DE CORRECCIÓN Z (ALTURA) ===
# ==========================================================

def calcular_correccion_z(y_referencia, y_borde, y_mitad):
    """
    Calcula la corrección de altura (Eje Z) en centésimas de milímetro (cMM).
    """
    global D_REAL_MM, OFFSET_CERO_PX

    delta_p_escala = abs(y_borde - y_mitad) 
    
    if delta_p_escala == 0 or D_REAL_MM == 0:
        return 0, "No se pudo calcular la escala Z (Etiquetas 'borde' y 'mitad' colapsaron)."
        
    factor_escala_px_mm = delta_p_escala / D_REAL_MM
    delta_p_bruto = y_borde - y_referencia
    delta_p_error = delta_p_bruto - OFFSET_CERO_PX
    correccion_cmm = (delta_p_error / factor_escala_px_mm) * 10 
    
    return int(round(correccion_cmm)), None

# ==========================================================
# === 3. FUNCIÓN INFERENCIA LATERAL (SEGURIDAD Y Z) (CORREGIDA) ===
# ==========================================================

def ejecutar_inferencia_lateral(model_lat, frame):
    """Ejecuta inferencia en la cámara lateral (SEGURIDAD Y CORRECCIÓN Z)."""
    global CLASE_REFERENCIA, CLASE_BORDE_ENV, CLASE_MITAD_ENV, CONFIDENCE_LAT

    results = model_lat.predict(source=frame, conf=CONFIDENCE_LAT, verbose=False) 
    annotated_lat = results[0].plot()
    
    response_code = CODIGO_RESPUESTA_OK
    correccion_z_cmm = 0 
    log_z = ""
    log_z_ref = "" # <--- CORRECCIÓN CLAVE: INICIALIZAR para evitar el error 'log_z_ref'
    
    y_coords = {CLASE_REFERENCIA: None, CLASE_BORDE_ENV: None, CLASE_MITAD_ENV: None}
    y_center_ref_fallback = frame.shape[0] // 2 if frame is not None else None
    
    # --- BÚSQUEDA DE DETECCIONES Y ANOMALÍAS ---
    for box in results[0].boxes:
        cls_name = model_lat.names.get(int(box.cls.item()))
        
        # 1. Búsqueda de Coordenadas Y para Z
        if cls_name in y_coords:
            y_center = int((box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2)
            y_coords[cls_name] = y_center
            
        # 2. Evaluación de Anomalías Críticas (PARADA)
        if cls_name in CLASES_ANOMALIA_LATERAL:
            print(f"🚨 Anomalía Lateral Crítica: {cls_name} detectada.")
            response_code = CODIGO_RESPUESTA_PARADA
            break 
            
    # 3. CÁLCULO DE CORRECCIÓN Z (SOLO SI NO HAY PARADA CRÍTICA)
    if response_code != CODIGO_RESPUESTA_PARADA:
        
        # LÓGICA FALLBACK Z: Si referencia_fija falla, usa el centro del frame
        if y_coords[CLASE_REFERENCIA] is None and y_center_ref_fallback is not None:
             y_coords[CLASE_REFERENCIA] = y_center_ref_fallback
             log_z_ref = "Usando centro de imagen como Referencia Z (Fallback)."

        if all(y_coords.values()):
            correccion_z_cmm, log_error = calcular_correccion_z(
                y_coords[CLASE_REFERENCIA], 
                y_coords[CLASE_BORDE_ENV], 
                y_coords[CLASE_MITAD_ENV]
            )
            if log_error:
                log_z = log_error
                correccion_z_cmm = 0
            else:
                log_z = f"📐 Cálculo Z exitoso."
        else:
            log_z = f"⚠️ Advertencia: No se detectaron las etiquetas críticas Z. Z=0."
            
    log_final = log_z + (f" ({log_z_ref})" if log_z_ref else "")
    
    return response_code, annotated_lat, correccion_z_cmm, log_final


# ==========================================================
# === 4. FUNCIÓN INFERENCIA SUPERIOR (QC, CONTEO Y Y) ===
# ==========================================================

def ejecutar_inferencia_superior(model_sup, frame):
    """
    Ejecuta la detección superior: 
    1. Evalúa Fallo de Calidad (QC).
    2. Cuenta las filas restantes y encuentra la Columna de Trabajo.
    3. Calcula la Corrección Y Dinámica.
    """
    global CLASE_VACIO, CLASE_POSICION, CLASE_ABANICO_Y, TOTAL_POSICIONES
    global TOLERANCIA_COLUMNA_PX, CORRECCION_Y_FIJA_PX, X_CENTROS_IDEALES, CONFIDENCE_SUP
    
    # Usa el umbral alto para reducir el ruido
    results = model_sup.predict(source=frame, conf=CONFIDENCE_SUP, verbose=False)
    annotated_sup = results[0].plot()
    
    has_qc_error = False
    detecciones_por_posicion = {} 
    
    for box in results[0].boxes:
        cls_name = model_sup.names.get(int(box.cls.item()))
        x_center = int((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2)

        if cls_name in CLASES_FALLO_SUPERIOR:
            has_qc_error = True
            
        if cls_name == CLASE_VACIO:
            detecciones_por_posicion[x_center] = 'VACIO'
        elif cls_name == CLASE_POSICION:
            if x_center not in detecciones_por_posicion:
                detecciones_por_posicion[x_center] = 'PRODUCTO'
        
    # --- CÁLCULO DE CONTEO ---
    # --- CÁLCULO DE CONTEO (LÓGICA CORREGIDA) ---
    
    # 1. Contar cuántas posiciones tienen PRODUCTO
    conteo_filas_restantes = 0
    posicion_x_trabajo = None 
    
    # Ordenamos las detecciones de izquierda a derecha (por posición X)
    posiciones_ordenadas = sorted(detecciones_por_posicion.keys())

    for x_pos in posiciones_ordenadas:
        estado = detecciones_por_posicion[x_pos]
        
        if estado == 'PRODUCTO':
            conteo_filas_restantes += 1
            # La columna de trabajo es la primera con producto, de izquierda a derecha
            if posicion_x_trabajo is None:
                posicion_x_trabajo = x_pos
                
    # Si la pinza siempre va a la columna 1 (más a la izquierda), entonces posicion_x_trabajo
    # ya está bien definido como la primera que encontró PRODUCTO.
    # Si la pinza toma la última (más a la derecha), necesitaríamos cambiar 'if posicion_x_trabajo is None' 
    # por un seguimiento y usar la última de las que tienen producto.
    
    # Asumiremos: La pinza siempre toma la columna más a la izquierda que NO esté VACIA.
    
    if conteo_filas_restantes == 0:
        conteo_filas_restantes = 0
    elif conteo_filas_restantes < 8:
        # Aquí, conteo_filas_restantes ya tiene el número correcto.
        pass # No hacemos nada, el conteo ya está bien.
    else:
        conteo_filas_restantes = TOTAL_POSICIONES # 8
        
    # El resto de la función (Corrección Y) sigue usando `posicion_x_trabajo`
    
    # --- CORRECCIÓN Y DINÁMICA ---
    correccion_y_pixels = 0
    
    if posicion_x_trabajo is not None and X_CENTROS_IDEALES:
        columna_actual = None
        min_dist = float('inf')
        
        # Paso 1: Determinar la columna de trabajo
        for num_col, x_ideal in X_CENTROS_IDEALES.items():
            dist = abs(posicion_x_trabajo - x_ideal)
            if dist < min_dist:
                min_dist = dist
                columna_actual = num_col
        
        # Paso 2: Aplicar Corrección Y si la desviación es significativa
        if columna_actual is not None and min_dist > TOLERANCIA_COLUMNA_PX:
             # Corrección Y es la diferencia del centro detectado al centro ideal (puede ser +/-)
             correccion_y_pixels = posicion_x_trabajo - X_CENTROS_IDEALES[columna_actual]
             
             # Limitar la corrección al valor fijo máximo
             correccion_y_pixels = max(min(correccion_y_pixels, CORRECCION_Y_FIJA_PX), -CORRECCION_Y_FIJA_PX)
             
             response_code = CODIGO_RESPUESTA_FALLO_QC # Activa QC por corrección Y
        else:
            # Si está alineado
            response_code = CODIGO_RESPUESTA_FALLO_QC if has_qc_error else CODIGO_RESPUESTA_OK
    else:
         # Si no se detectó envase o no se pudo calibrar X_CENTROS_IDEALES
         response_code = CODIGO_RESPUESTA_FALLO_QC if has_qc_error else CODIGO_RESPUESTA_OK

    return response_code, annotated_sup, conteo_filas_restantes, correccion_y_pixels


# ==========================================================
# === 5. BUCLE PRINCIPAL DE SIMULACIÓN (SIN PLC) ===
# ==========================================================

def simulacion_deteccion_video(video_sup_path, video_lat_path):
    """Bucle principal para probar la detección y cálculo con videos."""
    
    # FASE 1: INICIALIZACIÓN
    model_sup, model_lat = cargar_modelos(MODEL_SUP_PATH, MODEL_LAT_PATH)
    if not model_sup or not model_lat: return
    
    cap_sup, cap_lat = inicializar_entradas(video_sup_path, video_lat_path)
    if not cap_sup or not cap_lat: return

    # --- CALIBRACIÓN Y (DINÁMICA) ---
    ret_sup, frame_sup_calib = cap_sup.read()
    
    if ret_sup:
        calcular_centros_ideales(model_sup, frame_sup_calib) 
        cap_sup.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reposicionar el video al inicio
    else:
        print("❌ Error: No se pudo leer el primer frame del video Superior para calibración Y.")
        return
    # ---------------------------------

    print("\n--- SISTEMA DE VISIÓN EN MODO SIMULACIÓN (Leyendo videos) ---\n")
    frame_counter = 0

    try:
        # Bucle: Mientras ambos videos tengan frames disponibles
        while cap_sup.isOpened() and cap_lat.isOpened():
            
            # A. CAPTURA DE FRAMES
            ret_lat, frame_lat = cap_lat.read()
            ret_sup, frame_sup = cap_sup.read()
            
            if not ret_lat or not ret_sup:
                print("--- 🏁 Videos finalizados o error de lectura. Fin de la simulación. ---")
                break

            frame_counter += 1
            print(f"\n--- 💡 PROCESANDO FRAME {frame_counter} ---")

            # B. INFERENCIA LATERAL (SEGURIDAD Y CORRECCIÓN Z)
            response_code, annotated_lat, correccion_z, log_z = ejecutar_inferencia_lateral(model_lat, frame_lat)
            
            # C. INFERENCIA SUPERIOR (CALIDAD, CONTEO Y CORRECCIÓN Y)
            qc_code, annotated_sup, conteo, correccion_y = \
                ejecutar_inferencia_superior(model_sup, frame_sup)
            
            # --- IMPRESIÓN DE RESULTADOS (REEMPLAZO DEL PLC) ---
            
            # Diagnóstico General: Da prioridad a la parada crítica (lateral)
            if response_code == CODIGO_RESPUESTA_PARADA:
                diagnostico_general = "🛑 PARADA CRÍTICA (Lateral)!"
            elif qc_code == CODIGO_RESPUESTA_FALLO_QC:
                diagnostico_general = "⚠️ FALLO QC / Corrección Y requerida"
            else:
                diagnostico_general = "✅ OK - Listo para el retiro"
            
            print("=========================================================")
            print(f"| DIAGNÓSTICO GENERAL: {diagnostico_general}")
            print("---------------------------------------------------------")
            print(f"| Resultado Lateral Z: {log_z}")
            print(f"| Corrección Z (Apriete, cálculo): {correccion_z} cMM")
            print("---------------------------------------------------------")
            print(f"| Cont. Filas Restantes (Superior): {conteo}")
            print(f"| Corrección Y (Desvío Dinámico, píxeles): {correccion_y} px")
            print("=========================================================")


            # Desplegar frames
            cv2.imshow("Lateral - Deteccion/Z", annotated_lat)
            cv2.imshow("Superior - Conteo/Y/QC", annotated_sup)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"🚨 ERROR EN BUCLE PRINCIPAL: {e}")
        
    finally:
        # Cierre seguro de recursos
        print("\n--- CERRANDO SISTEMA DE VISIÓN DE SIMULACIÓN ---")
        if 'cap_sup' in locals() and cap_sup.isOpened(): cap_sup.release()
        if 'cap_lat' in locals() and cap_lat.isOpened(): cap_lat.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    simulacion_deteccion_video(VIDEO_SUP_PATH, VIDEO_LAT_PATH)