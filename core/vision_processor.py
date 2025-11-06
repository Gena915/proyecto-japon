"""
VisionProcessor - Módulo de procesamiento de visión artificial
Integra YOLO con el sistema de control PLC
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO


class VisionProcessor:
    """
    Procesador de visión artificial para el sistema PLC-YOLO (DUAL CAM).
    
    Responsabilidades:
    - Cargar dos modelos (Superior y Lateral)
    - Procesar resultados de YOLO de ambos frames
    - Calcular métricas (desviación, número de filas)
    - Filtrar detecciones por confianza
    - Aplicar calibración espacial (mm/pixel)
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                modelo_path_sup: str = None,  # Nueva ruta para Superior
                modelo_path_lat: str = None): # Nueva ruta para Lateral
        """
        Inicializa el procesador de visión para dos cámaras, cargando dos confianzas.
        """
        # Asume que el diccionario 'config' pasado contiene 'vision_sup' y 'vision_lat'
        self.config_sup = config.get('vision_sup', {})
        self.config_lat = config.get('vision_lat', {})
        
        # --- Configuración Superior (Y, QC, Conteo) ---
        self.mm_per_pixel_sup = self.config_sup.get('mm_per_pixel')
        # CAMBIO CLAVE: Confianza Superior
        self.confianza_minima_sup = self.config_sup.get('confidence_sup', 0.60) 
        self.usar_centro_imagen_sup = self.config_sup.get('usar_centro_imagen')
        self.referencia_x_custom_sup = self.config_sup.get('referencia_x_custom')

        # --- Configuración Lateral (Z, Seguridad) ---
        self.mm_per_pixel_lat = self.config_lat.get('mm_per_pixel')
        # CAMBIO CLAVE: Confianza Lateral
        self.confianza_minima_lat = self.config_lat.get('confidence_lat', 0.05)
        self.usar_centro_imagen_lat = self.config_lat.get('usar_centro_imagen') # Podrías necesitar esto en Z
        self.referencia_x_custom_lat = self.config_lat.get('referencia_x_custom')
        
        # --- Modelos ---
        self.model_sup = None
        self.model_lat = None
        self.modelos_cargados = False
        
        if modelo_path_sup and modelo_path_lat:
            self.cargar_modelos(modelo_path_sup, modelo_path_lat)
            
    # La lógica de cargar_modelo original se convierte en cargar_modelos_dual
    def cargar_modelos(self, path_sup: str, path_lat: str) -> bool:
        """
        Carga ambos modelos YOLO desde archivo.
        """
        try:
            print(f"📦 Cargando modelo SUPERIOR YOLO desde {path_sup}...")
            self.model_sup = YOLO(path_sup)
            print(f"📦 Cargando modelo LATERAL YOLO desde {path_lat}...")
            self.model_lat = YOLO(path_lat)
            print("✅ Ambos modelos YOLO cargados exitosamente")
            self.modelos_cargados = True
            return True
        except Exception as e:
            print(f"❌ Error cargando uno o ambos modelos: {e}")
            self.modelos_cargados = False
            return False
        
    def procesar_frames_dual(self, frame_sup, frame_lat) -> Tuple[Dict, Dict]:
        """
        Ejecuta la inferencia de YOLO para ambas cámaras y procesa sus resultados.
        """
        if not self.modelos_cargados:
            print("❌ Error: Modelos no cargados. Abortando procesamiento dual.")
            # Devuelve respuestas de fallo para ambos
            fallo_sup = self._generar_respuesta_fallo("Modelos no cargados (SUP)")
            fallo_lat = self._generar_respuesta_fallo("Modelos no cargados (LAT)")
            return fallo_sup, fallo_lat

        # 1. INFERENCIA Y PROCESAMIENTO SUPERIOR (QC, Y)
        results_sup = self.model_sup.predict(
            source=frame_sup, 
            conf=self.confianza_minima_sup, # <-- CONFIDENCIA SUPERIOR USADA AQUÍ
            verbose=False
        )
        data_sup = self._procesar_yolo_detecciones(
            yolo_results=results_sup, 
            ancho_imagen=frame_sup.shape[1], 
            alto_imagen=frame_sup.shape[0], 
            confianza_umbral=self.confianza_minima_sup, # <-- PASA LA CONFIDENCIA
            camara_tipo="SUP"
        )

        # 2. INFERENCIA Y PROCESAMIENTO LATERAL (Seguridad, Z)
        results_lat = self.model_lat.predict(
            source=frame_lat, 
            conf=self.confianza_minima_lat, # <-- CONFIDENCIA LATERAL USADA AQUÍ
            verbose=False
        )
        data_lat = self._procesar_yolo_detecciones(
            yolo_results=results_lat, 
            ancho_imagen=frame_lat.shape[1], 
            alto_imagen=frame_lat.shape[0], 
            confianza_umbral=self.confianza_minima_lat, # <-- PASA LA CONFIDENCIA
            camara_tipo="LAT"
        )
        
        return data_sup, data_lat
    
    def _procesar_yolo_detecciones(self, # <--- ¡MÉTODO RENOMBRADO!
                                   yolo_results,
                                   ancho_imagen: int,
                                   alto_imagen: int,
                                   confianza_umbral: float, # <--- NUEVO ARGUMENTO CLAVE
                                   camara_tipo: str) -> Dict: # <--- NUEVO ARGUMENTO CLAVE
        """
        Procesa resultados de YOLO y calcula métricas, usando el umbral de confianza provisto.
        """
        result = yolo_results[0]
        
        # Validación: sin detecciones
        if result.boxes is None or len(result.boxes) == 0:
            return self._generar_respuesta_fallo(
                f"No se detectaron objetos en la imagen ({camara_tipo})"
            )
        
        # Filtrar detecciones válidas
        # Llama a la versión de _filtrar_por_confianza que acepta el umbral
        detecciones_validas = self._filtrar_por_confianza(
            boxes=result.boxes,
            confianza_umbral=confianza_umbral 
        )
        
        if len(detecciones_validas) == 0:
            return self._generar_respuesta_fallo(
                f"Ninguna detección supera el umbral de confianza ({confianza_umbral*100:.0f}%) en la cámara {camara_tipo}"
            )
        
        # Calcular métricas
        num_filas = len(detecciones_validas)
        desviacion_mm = self._calcular_desviacion(
            detecciones_validas, 
            ancho_imagen
        )
        
        # Generar metadata
        confianzas = [d['confianza'] for d in detecciones_validas]
        
        return {
            'success': True,
            'filas': num_filas,
            'desviacion_mm': desviacion_mm,
            'metadata': {
                'camara_tipo': camara_tipo,
                'confianza_umbral_usado': confianza_umbral,
                'total_detectado': len(result.boxes),
                'detecciones_validas': len(detecciones_validas),
                'confianza_promedio': float(np.mean(confianzas)),
                'confianza_minima_detectada': float(np.min(confianzas)),
                'confianza_maxima': float(np.max(confianzas)),
                'ancho_imagen': ancho_imagen,
                'alto_imagen': alto_imagen
            }
        }
        
        
    
    def _filtrar_por_confianza(self, boxes, confianza_umbral: float) -> List[Dict]:
        """
        Filtra cajas de detección por umbral de confianza, recibiendo el umbral
        como argumento para ser usado por la cámara Superior o Lateral.
        """
        detecciones = []
        
        for box in boxes:
            confianza = float(box.conf[0].item())
            
            # Usa el umbral que se le pasó (sup o lat)
            if confianza >= confianza_umbral: 
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detecciones.append({
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'ancho': x2 - x1,
                    'alto': y2 - y1,
                    'confianza': confianza,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detecciones
    
    def _calcular_desviacion(self, 
                            detecciones: List[Dict], 
                            ancho_imagen: int) -> float:
        """
        Calcula la desviación en milímetros.
        
        Estrategia: Encontrar el objeto más cercano al punto de referencia.
        
        Args:
            detecciones: Lista de diccionarios con detecciones
            ancho_imagen: Ancho de la imagen en píxeles
            
        Returns:
            Desviación en mm (positivo=derecha, negativo=izquierda)
        """
        # Determinar punto de referencia
        if self.usar_centro_imagen or self.referencia_x_custom is None:
            punto_referencia = ancho_imagen / 2
        else:
            punto_referencia = self.referencia_x_custom
        
        # Buscar objeto más cercano al punto de referencia
        min_distancia_abs = float('inf')
        desviacion_objetivo = 0.0
        
        for det in detecciones:
            desviacion_px = det['center_x'] - punto_referencia
            distancia_abs = abs(desviacion_px)
            
            if distancia_abs < min_distancia_abs:
                min_distancia_abs = distancia_abs
                desviacion_objetivo = desviacion_px
        
        # Convertir a mm
        desviacion_mm = desviacion_objetivo * self.mm_per_pixel
        
        return desviacion_mm
    
    def _generar_respuesta_fallo(self, razon: str) -> Dict:
        """
        Genera respuesta estructurada para casos de fallo.
        
        Args:
            razon: Descripción del fallo
            
        Returns:
            Dict con success=False y metadata
        """
        print(f"⚠️ Procesamiento fallido: {razon}")
        
        return {
            'success': False,
            'filas': 0,
            'desviacion_mm': 0.0,
            'metadata': {
                'razon_fallo': razon
            }
        }
    
    def validar_resultado(self, resultado: Dict) -> Tuple[bool, List[str]]:
        """
        Valida un resultado antes de enviarlo al PLC.
        
        Validaciones:
        - Rango razonable de desviación
        - Número de filas lógico
        - Coherencia de datos
        
        Args:
            resultado: Diccionario retornado por procesar_resultados()
            
        Returns:
            (es_valido, lista_de_advertencias)
        """
        advertencias = []
        
        if not resultado['success']:
            return True, []  # Los fallos no necesitan validación adicional
        
        # Validar desviación
        desv = abs(resultado['desviacion_mm'])
        if desv > 500:  # >50cm es sospechoso
            advertencias.append(
                f"⚠️ Desviación muy grande: {resultado['desviacion_mm']:.2f}mm"
            )
        
        # Validar número de filas
        filas = resultado['filas']
        if filas < 0 or filas > 100:
            advertencias.append(f"⚠️ Número de filas inusual: {filas}")
        
        # Coherencia
        if filas == 0 and desv > 0:
            advertencias.append("⚠️ Incoherencia: 0 filas pero desviación != 0")
        
        return len(advertencias) == 0, advertencias
    
    def ajustar_calibracion(self, nuevo_mm_per_pixel: float) -> None:
        """
        Ajusta la calibración espacial del sistema.
        
        Args:
            nuevo_mm_per_pixel: Nueva relación mm/píxel
        """
        self.mm_per_pixel = nuevo_mm_per_pixel
        print(f"🔧 Calibración actualizada: {self.mm_per_pixel} mm/píxel")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Configuración de ejemplo
    config_ejemplo = {
        'vision': {
            'mm_per_pixel': 0.5,
            'confianza_minima': 0.5,
            'usar_centro_imagen': True,
            'referencia_x_custom': None
        }
    }
    
    processor = VisionProcessor(config_ejemplo)
    print("✅ VisionProcessor inicializado")