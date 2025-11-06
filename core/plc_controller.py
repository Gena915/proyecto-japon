"""
PLCController - Módulo de comunicación con PLC Mitsubishi
Implementa el protocolo MC Type3E y el handshake de control
"""

import pymcprotocol
import json
from typing import Optional, Dict, Tuple


class PLCController:
    """
    Controlador para comunicación con PLC Mitsubishi via MC Protocol.
    
    Responsabilidades:
    - Gestionar conexión TCP/IP con el PLC
    - Implementar protocolo de handshake (D28: 99→88/77)
    - Codificar/decodificar datos (mm → int32, etc.)
    - Manejar reconexiones automáticas
    """
    
    def __init__(self, config_file: str = 'config/plc_config.json'):
        """
        Inicializa el controlador con configuración desde JSON.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        self.config = self._cargar_configuracion(config_file)
        self.mc = None
        self.is_connected = False
        
        # Extraer configuraciones
        conn = self.config.get('conexion', {})
        dirs = self.config.get('direcciones', {})
        codigos = self.config.get('codigos_estado', {})
        
        self.ip_plc = conn.get('ip_plc', '127.0.0.1')
        self.puerto_plc = conn.get('puerto_plc', 5007)
        
        self.DEV_TRIGGER = dirs.get('dispositivo_trigger', 'D701')  # <--- MODIFICADO (D28 -> D701)
        self.DEV_RESULTADO_VALOR = dirs.get('dispositivo_valor', 'D710')  # <--- MODIFICADO (D29 -> D710)
        self.DEV_RESULTADO_FILAS = dirs.get('dispositivo_filas', 'D714')  # <--- MODIFICADO (D14 -> D714)
        self.DEV_RESULTADO_VALOR_Z = dirs.get('dispositivo_valor_z', 'D712')  # <--- MODIFICADO (D31 -> D712)
        
        self.VAL_SOLICITUD = codigos.get('valor_solicitud', 99)
        self.VAL_EXITO = codigos.get('valor_exito', 88)
        self.VAL_ERROR = codigos.get('valor_error', 77)
    
    def _cargar_configuracion(self, config_file: str) -> Dict:
        """Carga configuración desde archivo JSON"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuración cargada desde {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Archivo {config_file} no encontrado, usando valores por defecto")
            return self._configuracion_por_defecto()
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON: {e}")
            raise
    
    def _configuracion_por_defecto(self) -> Dict:
        """Retorna configuración por defecto si falla la carga"""
        return {
            "conexion": {
                "ip_plc": "192.168.100.120",
                "puerto_plc": 5007
            },
            "direcciones": {
                "dispositivo_trigger": "D701",  # <--- MODIFICADO
                "dispositivo_valor": "D710",    # Desviación Y (32 bits, D710, D711) - MODIFICADO
                "dispositivo_filas": "D714",    # Número de Filas (16 bits) - MODIFICADO
                "dispositivo_valor_z": "D712"   # Corrección Z (32 bits, D712, D713) - MODIFICADO
            },
            "codigos_estado": {
                "valor_solicitud": 99,
                "valor_exito": 88,
                "valor_error": 77
            }
        }
    
    def conectar(self) -> bool:
        """
        Establece conexión con el PLC.
        
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        print(f"🔌 Conectando al PLC en {self.ip_plc}:{self.puerto_plc}...")
        try:
            self.mc = pymcprotocol.Type3E()
            self.mc.connect(self.ip_plc, self.puerto_plc)
            self.is_connected = True
            print("✅ Conexión PLC establecida exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error al conectar con PLC: {e}")
            self.is_connected = False
            return False
    
    def desconectar(self) -> None:
        """Cierra la conexión con el PLC de forma segura"""
        if self.is_connected and self.mc:
            try:
                self.mc.close()
                print("✅ Desconectado del PLC")
            except Exception as e:
                print(f"⚠️ Error al desconectar: {e}")
            finally:
                self.is_connected = False
                self.mc = None
    
    def leer_solicitud_inspeccion(self) -> bool:
        """
        Lee el registro D28 para verificar si hay solicitud de inspección.
        
        Protocolo:
        - D28 = 99: PLC solicita inspección
        
        Returns:
            True si D28 == 99, False en caso contrario
        """
        if not self.is_connected:
            return False
        
        try:
            valor = self.mc.batchread_wordunits(
                headdevice=self.DEV_TRIGGER, 
                readsize=1
            )[0]
            
            if valor == self.VAL_SOLICITUD:
                print(f"📥 Solicitud de inspección detectada ({self.DEV_TRIGGER}={self.VAL_SOLICITUD})")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error al leer {self.DEV_TRIGGER}: {e}")
            self.is_connected = False
            return False
    
    def escribir_resultados(self, 
                            desviacion_y_mm: float, 
                            num_filas: int, 
                            correccion_z_mm: float, 
                            codigo_respuesta: int) -> bool:
        """
        Escribe los resultados de la inspección al PLC (Dual Cam).
        
        Protocolo de escritura (orden crítico):
        1. D29 (desviación Y en 1/100 mm, 32 bits)
        2. D31 (corrección Z en 1/100 mm, 32 bits)
        3. D14 (número de filas, 16 bits)
        4. D28 (estado: 88=éxito, 77=error/parada)
        
        Args:
            desviacion_y_mm: Desviación en Y (Horizontal) en milímetros (float)
            num_filas: Número de filas detectadas (int)
            correccion_z_mm: Corrección en Z (Profundidad) en milímetros (float)
            codigo_respuesta: Código final a enviar (88 para OK, 77 para ERROR/Parada)
            
        Returns:
            True si la escritura fue exitosa
        """
        if not self.is_connected:
            print("❌ No se puede escribir: sin conexión PLC")
            return False
        
        try:
            # 1. Convertir Desviación Y (D29, D30)
            valor_desviacion_y = int(round(desviacion_y_mm * 100.0))
            palabras_valor_y = self._int32_to_words(valor_desviacion_y)
            
            # 2. Convertir Corrección Z (D31, D32)
            valor_correccion_z = int(round(correccion_z_mm * 100.0))
            palabras_valor_z = self._int32_to_words(valor_correccion_z)

            # 3. Validar número de filas (D14)
            valor_filas = max(0, int(num_filas))
            
            # ORDEN CRÍTICO: Escribir D29, D31, D14, luego el estado D28
            
            # Escribir Desviación Y (D29)
            self.mc.batchwrite_wordunits(
                headdevice=self.DEV_RESULTADO_VALOR, 
                values=palabras_valor_y
            )
            
            # Escribir Corrección Z (D31)
            self.mc.batchwrite_wordunits(
                headdevice=self.DEV_RESULTADO_VALOR_Z, 
                values=palabras_valor_z
            )
            
            # Escribir Filas (D14)
            self.mc.batchwrite_wordunits(
                headdevice=self.DEV_RESULTADO_FILAS, 
                values=[valor_filas]
            )
            
            # Escribir Código de Respuesta (D28)
            self.mc.batchwrite_wordunits(
                headdevice=self.DEV_TRIGGER, 
                values=[codigo_respuesta]
            )
            
            print(f"✅ Resultados DUALES enviados: Y_Desv={desviacion_y_mm:.2f}mm ({valor_desviacion_y}), "
                    f"Z_Corr={correccion_z_mm:.2f}mm ({valor_correccion_z}), "
                    f"Filas={valor_filas}, Estado={codigo_respuesta}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al escribir resultados: {e}")
            self.is_connected = False
            return False
    
    def _int32_to_words(self, n: int) -> list:
        """
        Convierte un entero con signo de 32 bits a dos palabras de 16 bits.
        
        Formato PLC: [low_word, high_word]
        """
        # Clamp al rango int32
        n = max(-2147483648, min(n, 2147483647))
        
        # Convertir a unsigned si es negativo
        if n < 0:
            n = n + (1 << 32)
        
        low_word = n & 0xFFFF
        high_word = (n >> 16) & 0xFFFF
        
        return [low_word, high_word]
    
    def verificar_conexion(self) -> bool:
        """
        Verifica si la conexión con el PLC sigue activa.
        
        Returns:
            True si la conexión está activa
        """
        if not self.is_connected or not self.mc:
            return False
        
        try:
            # Intenta leer el registro de trigger
            self.mc.batchread_wordunits(headdevice=self.DEV_TRIGGER, readsize=1)
            return True
        except Exception:
            self.is_connected = False
            return False
    
    def obtener_estado_sistema(self) -> Dict:
        """
        Lee el estado completo del sistema desde el PLC.
        
        Returns:
            Diccionario con estado actual de D28, D29, D14
        """
        if not self.is_connected:
            return {'conectado': False}
        
        try:
            trigger = self.mc.batchread_wordunits(headdevice=self.DEV_TRIGGER, readsize=1)[0]
            filas = self.mc.batchread_wordunits(headdevice=self.DEV_RESULTADO_FILAS, readsize=1)[0]
            
            return {
                'conectado': True,
                'trigger': trigger,
                'filas': filas,
                'descripcion_trigger': self._describir_codigo(trigger)
            }
        except Exception as e:
            print(f"⚠️ Error leyendo estado: {e}")
            return {'conectado': False, 'error': str(e)}
    
    def _describir_codigo(self, codigo: int) -> str:
        """Convierte código numérico a descripción legible"""
        if codigo == self.VAL_SOLICITUD:
            return "SOLICITUD PENDIENTE"
        elif codigo == self.VAL_EXITO:
            return "ÚLTIMA INSPECCIÓN: ÉXITO"
        elif codigo == self.VAL_ERROR:
            return "ÚLTIMA INSPECCIÓN: ERROR"
        elif codigo == 0:
            return "IDLE"
        else:
            return f"CÓDIGO DESCONOCIDO ({codigo})"


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Test básico del controlador
    plc = PLCController()
    
    if plc.conectar():
        print("\n📊 Estado del sistema:")
        estado = plc.obtener_estado_sistema()
        print(estado)
        
        plc.desconectar()