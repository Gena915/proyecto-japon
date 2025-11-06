import cv2
from ultralytics import YOLO
import time
import os
# IMPORTAR AQUÍ LA LIBRERÍA DE COMUNICACIÓN PLC (Ej: import pycomm3)

# ==========================================================
# === 0. CONFIGURACIÓN GENERAL Y RUTAS ===
# ==========================================================
MODEL_SUP_PATH = "modelos/modelo_superior_qc.pt"
MODEL_LAT_PATH = "modelos/modelo_lateral_anomalias.pt"
CAM_SUP_ID = 0 
CAM_LAT_ID = 1 
CONFIDENCE_THRESHOLD = 0.5 

# Clases de Visión (DEBE COINCIDIR CON TU ENTRENAMIENTO)
CLASES_ANOMALIA_LATERAL = ['posicion_correcta', 'error_caido'] 
CLASES_FALLO_SUPERIOR = ['error_apilado', 'posicion_vacia', 'error_alerta', 'error_abanico', 'posicion_columna'] # Se quita 'desalineacion_y_abanico' para manejarlo aparte
CLASE_ABANICO_Y = 'desalineacion_y_abanico' # Nueva constante para la corrección Y

# Clases necesarias para la nueva lógica:
CLASE_POSICION = 'posicion_columna' # Existencia de columna (independiente de QC)
CLASE_VACIO = 'posicion_vacia'       # Espacio donde se retiró la columna
TOTAL_POSICIONES = 8                 # Máximo de filas en un nivel

# --- Constantes para Corrección Z ---
CLASE_REFERENCIA = 'referencia_fija'
CLASE_BORDE_ENV = 'borde_envase'
CLASE_MITAD_ENV = 'mitad_envase'

# Valores de Calibración
D_REAL_MM = 100.0 # Distancia real conocida entre Borde y Mitad (H/2)
OFFSET_CERO_PX = 40 # Offset de píxeles cuando el error es 0 mm (!!! AJUSTAR EN CAMPO !!!)
CORRECCION_Y_FIJA_PX = 50 # Valor fijo de corrección Y (!!! AJUSTAR EN CAMPO !!!)
TOLERANCIA_COLUMNA_PX = 50 # 🚨 NUEVO: Rango aceptable para filtrar abanico en Columna de Trabajo

# Códigos de Comunicación PLC (Ejemplo)
CODIGO_PETICION_VISION = 99
CODIGO_RESPUESTA_OK = 0
CODIGO_RESPUESTA_FALLO_QC = 1 # Fallo de Calidad O Corrección Y requerida (Superior)
CODIGO_RESPUESTA_PARADA = 2 # Fallo Grave (Lateral)

# --- Registros PLC (Simulación) ---
REGISTRO_RESPUESTA = "DB_VIS.Respuesta"
REGISTRO_CONTEO = "DB_VIS.FilasRestantes"
REGISTRO_CORRECCION_Y = "DB_VIS.CorreccionY"
REGISTRO_CORRECCION_Z = "DB_VIS.CorreccionZ"
# REGISTRO_NIVEL_ACTUAL = "DB_VIS.NivelActual" # 🚨 FUTURO: Registro para distinguir 3 niveles

# ==========================================================
# === 1. FUNCIONES SIMULADAS (REEMPLAZAR POR LÓGICA PLC REAL) ===
# ==========================================================

def simular_lectura_plc(registro):
    """Simula la lectura del registro del PLC."""
    # Lógica de simulación para activar un ciclo y luego mantenerlo en 0
    if not hasattr(simular_lectura_plc, 'ciclo_activo'):
        simular_lectura_plc.ciclo_activo = True
        return CODIGO_PETICION_VISION
    if simular_lectura_plc.ciclo_activo:
        time.sleep(1)
        simular_lectura_plc.ciclo_activo = False
        return 0
    return 0

def simular_escritura_plc(registro, valor):
    """Simula la escritura de la respuesta al PLC."""
    print(f"🤖 PLC WRITE: Registro {registro} <- Valor {valor}")
    time.sleep(0.01)
    pass


# ==========================================================
# === 2. FUNCIONES DE HARDWARE Y VISIÓN (Inicialización) ===
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

def inicializar_camaras(id_sup, id_lat):
    """Inicializa las cámaras de video."""
    cap_sup = cv2.VideoCapture(id_sup)
    cap_lat = cv2.VideoCapture(id_lat)
    if not cap_sup.isOpened():
        print(f"❌ ERROR: No se puede abrir la Cámara Superior (ID: {id_sup})")
        return None, None
    if not cap_lat.isOpened():
        print(f"❌ ERROR: No se puede abrir la Cámara Lateral (ID: {id_lat})")
        return None, None
    return cap_sup, cap_lat

def tomar_frame(cap):
    """Captura un solo frame de la cámara."""
    ret, frame = cap.read()
    if ret:
        return frame
    return None

# ==========================================================
# === 3. FUNCIÓN: CÁLCULO DE CORRECCIÓN Z (ALTURA) ===
# ==========================================================

# La función usa las constantes globales D_REAL_MM y OFFSET_CERO_PX
def calcular_correccion_z(y_referencia, y_borde, y_mitad):
    """
    Calcula la corrección de altura (Eje Z) en centésimas de milímetro (cMM).
    """
    global D_REAL_MM, OFFSET_CERO_PX

    # PASO 1: ESCALA DINÁMICA (C_p/mm)
    delta_p_escala = abs(y_borde - y_mitad) 
    
    if delta_p_escala == 0 or D_REAL_MM == 0:
        return 0
        
    factor_escala_px_mm = delta_p_escala / D_REAL_MM
    
    # PASO 2: ERROR NETO EN PÍXELES
    # Error Bruto: Medición actual entre el envase y la referencia fija.
    delta_p_bruto = y_borde - y_referencia
    
    # Error Neto: Error Bruto ajustado por el Offset Cero.
    delta_p_error = delta_p_bruto - OFFSET_CERO_PX
    
    # PASO 3: CONVERSIÓN Y SALIDA
    # Multiplicar por 10 para obtener Centésimas de Milímetro (cMM).
    # El signo indica la dirección de corrección (por ejemplo, Z positiva si el envase está muy bajo)
    correccion_cmm = (delta_p_error / factor_escala_px_mm) * 10 
    
    return int(round(correccion_cmm))

# ==========================================================
# === 4. FUNCIÓN INFERENCIA LATERAL (SEGURIDAD Y Z) ===
# ==========================================================

def ejecutar_inferencia_lateral(model_lat, frame):
    """Ejecuta inferencia en la cámara lateral (SEGURIDAD Y CORRECCIÓN Z)."""
    global CLASE_REFERENCIA, CLASE_BORDE_ENV, CLASE_MITAD_ENV, CONFIDENCE_THRESHOLD

    results = model_lat.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated_lat = results[0].plot()
    
    response_code = CODIGO_RESPUESTA_OK
    correccion_z_cmm = None # Inicializa la corrección Z

    # Variables para Corrección Z
    y_coords = {CLASE_REFERENCIA: None, CLASE_BORDE_ENV: None, CLASE_MITAD_ENV: None}
    
    for box in results[0].boxes:
        cls_name = model_lat.names.get(int(box.cls.item()))
        
        # 1. Búsqueda de Coordenadas Y para Z
        if cls_name in y_coords:
            # Tomamos el centro Y del Bounding Box
            y_center = int((box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2)
            y_coords[cls_name] = y_center
            
        # 2. Evaluación de Anomalías Críticas (PARADA)
        if cls_name in CLASES_ANOMALIA_LATERAL:
            print(f"🚨 Anomalía Lateral Crítica: {cls_name} detectada.")
            response_code = CODIGO_RESPUESTA_PARADA
            break # Si hay PARADA, el cálculo Z es irrelevante
            
    # 3. CÁLCULO DE CORRECCIÓN Z (Solo si no hay PARADA y se detectan las 3 etiquetas)
    if response_code != CODIGO_RESPUESTA_PARADA:
        if all(y_coords.values()):
            correccion_z_cmm = calcular_correccion_z(
                y_coords[CLASE_REFERENCIA], 
                y_coords[CLASE_BORDE_ENV], 
                y_coords[CLASE_MITAD_ENV]
            )
            print(f"📐 Corrección Z calculada: {correccion_z_cmm} cMM.")
        else:
            print("⚠️ Advertencia: No se detectaron las 3 etiquetas Z. Corrección Z no calculada.")
            
    return response_code, annotated_lat, correccion_z_cmm # DEVOLVEMOS Z


# ==========================================================
# === 5. FUNCIÓN INFERENCIA SUPERIOR (QC, CONTEO Y Y) ===
# ==========================================================

def ejecutar_inferencia_superior(model_sup, frame):
    """
    Ejecuta la detección superior: 
    1. Evalúa Fallo de Calidad (QC).
    2. Cuenta las filas restantes y encuentra la Columna de Trabajo.
    3. Filtra la Corrección Y para aplicarla solo a la Columna de Trabajo.
    """
    global CLASE_VACIO, CLASE_POSICION, CLASE_ABANICO_Y, TOTAL_POSICIONES
    global TOLERANCIA_COLUMNA_PX, CORRECCION_Y_FIJA_PX
    
    results = model_sup.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated_sup = results[0].plot()
    
    has_qc_error = False
    detecciones_por_posicion = {} # {x_center: 'VACIO'/'PRODUCTO'}
    abanico_x_centers = [] # Lista de centros X donde se detectó abanico
    
    
    for box in results[0].boxes:
        cls_name = model_sup.names.get(int(box.cls.item()))
        x_center = int((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2)

        # 1. EVALUACIÓN QC
        if cls_name in CLASES_FALLO_SUPERIOR:
            has_qc_error = True
            
        # 2. AGRUPACIÓN DE POSICIÓN (para el conteo)
        if cls_name == CLASE_VACIO:
            detecciones_por_posicion[x_center] = 'VACIO'
        elif cls_name == CLASE_POSICION:
            if x_center not in detecciones_por_posicion:
                detecciones_por_posicion[x_center] = 'PRODUCTO'
        
        # 3. REGISTRO DE ABANICO
        if cls_name == CLASE_ABANICO_Y:
            abanico_x_centers.append(x_center)
        
    # --- 4. CÁLCULO DE CONTEO Y COLUMNA DE TRABAJO ---
    
    # Obtener y ordenar las coordenadas X de las detecciones
    posiciones_ordenadas = sorted(detecciones_por_posicion.keys())
    
    posiciones_retiradas = 0
    posicion_x_trabajo = None 
    
    # Contar vacíos consecutivos y encontrar la primera columna de producto
    for x_pos in posiciones_ordenadas:
        if detecciones_por_posicion[x_pos] == 'VACIO':
            posiciones_retiradas += 1
        else:
            # La primera columna que NO es vacía es la COLUMNA DE TRABAJO
            if posicion_x_trabajo is None:
                posicion_x_trabajo = x_pos
            break 
            
    conteo_filas_restantes = TOTAL_POSICIONES - posiciones_retiradas
    
    # --- 5. CÁLCULO DE CORRECCIÓN Y (FILTRADO POR COLUMNA DE TRABAJO) ---
    correccion_y_pixels = 0
    requiere_correccion = False
    
    if posicion_x_trabajo is not None:
        for abanico_x in abanico_x_centers:
            # 🚨 FILTRO CRÍTICO: ¿El abanico detectado está cerca de la Columna de Trabajo?
            if abs(abanico_x - posicion_x_trabajo) < TOLERANCIA_COLUMNA_PX:
                requiere_correccion = True
                correccion_y_pixels = CORRECCION_Y_FIJA_PX # Valor fijo calibrado
                break 
    
    # --- 6. RESPUESTA FINAL ---
    if requiere_correccion or has_qc_error:
        response_code = CODIGO_RESPUESTA_FALLO_QC 
        if requiere_correccion:
             print(f"📐 Corrección Y ({correccion_y_pixels}px) requerida en la Columna de Trabajo.")
    else:
        response_code = CODIGO_RESPUESTA_OK

    return response_code, annotated_sup, conteo_filas_restantes, correccion_y_pixels


# ==========================================================
# === 6. BUCLE PRINCIPAL DE CONTROL MAESTRO ===
# ==========================================================

def control_maestro_produccion():
    """Bucle principal de espera de señal de PLC."""
    
    # FASE 1: INICIALIZACIÓN
    model_sup, model_lat = cargar_modelos(MODEL_SUP_PATH, MODEL_LAT_PATH)
    if not model_sup or not model_lat: return
    
    cap_sup, cap_lat = inicializar_camaras(CAM_SUP_ID, CAM_LAT_ID)
    if not cap_sup or not cap_lat: return

    print("\n--- SISTEMA DE VISIÓN EN MODO ESPERA (Ready) ---\n")
    
    try:
        while True:
            # Lectura del comando PLC
            comando_plc = simular_lectura_plc(REGISTRO_RESPUESTA)
            
            if comando_plc == CODIGO_PETICION_VISION:
                print("--- 💡 SEÑAL DE PLC RECIBIDA. INICIANDO CICLO ---")
                
                # A. CAPTURA DE FRAMES
                frame_lat = tomar_frame(cap_lat)
                frame_sup = tomar_frame(cap_sup)
                
                if frame_lat is None or frame_sup is None:
                    print("❌ ERROR DE CAPTURA. SALTANDO CICLO.")
                    simular_escritura_plc(REGISTRO_RESPUESTA, CODIGO_RESPUESTA_FALLO_QC)
                    continue
                
                # B. INFERENCIA LATERAL (PRIORIDAD: SEGURIDAD Y CORRECCIÓN Z)
                response_code, annotated_lat, correccion_z = ejecutar_inferencia_lateral(model_lat, frame_lat)
                
                if response_code == CODIGO_RESPUESTA_PARADA:
                    print("🛑 ERROR CRÍTICO DETECTADO. ENVIANDO PARADA.")
                    simular_escritura_plc(REGISTRO_RESPUESTA, CODIGO_RESPUESTA_PARADA)
                else:
                    # C. INFERENCIA SUPERIOR (CALIDAD, CONTEO Y CORRECCIÓN Y)
                    qc_code, annotated_sup, conteo, correccion_y = \
                        ejecutar_inferencia_superior(model_sup, frame_sup)
                    
                    # --- ESCRITURA DE DATOS AL PLC ---
                    simular_escritura_plc(REGISTRO_CONTEO, conteo)
                    simular_escritura_plc(REGISTRO_CORRECCION_Y, correccion_y)
                    
                    # ESCRITURA Z (Solo si se calculó un valor válido)
                    if correccion_z is not None:
                        simular_escritura_plc(REGISTRO_CORRECCION_Z, correccion_z) # <-- ESCRITURA Z

                    simular_escritura_plc(REGISTRO_RESPUESTA, qc_code) # Respuesta Final (OK o QC Fallo/Corrección)

                    print(f"✅ CICLO COMPLETADO. Respuesta: {qc_code} | Filas: {conteo} | Corrección Y: {correccion_y}px | Corrección Z: {correccion_z}cMM\n")
                
                # Desplegar frames
                cv2.imshow("Lateral - Anomalía (Con Z)", annotated_lat)
                cv2.imshow("Superior - Calidad", annotated_sup)
                cv2.waitKey(1)

            else:
                time.sleep(0.05) 
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"🚨 ERROR EN BUCLE PRINCIPAL: {e}")
        
    finally:
        # Cierre seguro de recursos
        print("\n--- CERRANDO SISTEMA DE VISIÓN ---")
        # Se agrega manejo de 'model_sup' y 'model_lat' si es necesario
        if 'cap_sup' in locals() and cap_sup.isOpened(): cap_sup.release()
        if 'cap_lat' in locals() and cap_lat.isOpened(): cap_lat.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Asegúrate de reemplazar las funciones de simulación por la conexión PLC real
    control_maestro_produccion()