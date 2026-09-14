# PARAMETROS DE VALIDACION

# Resolucion minima de la imagen
MIN_ANCHO = 100
MIN_ALTO = 100
# Iluminacion (brillo promedio del rostro recortado, escala 0-255)
MIN_BRILLO = 55
MAX_BRILLO = 205
# Nitidez (pendiente de implementar validar_nitidez)
MIN_NITIDEZ = 80
# Tamano minimo del rostro detectado, en pixeles
MIN_ANCHO_ROSTRO = 100
# Tamano del rostro respecto al total de la imagen (proporcion de area)
# REVISAR: bajado de 0.18 a 0.03 tras probar con fotos reales que rondaban 5-6%
MIN_TAMANO_ROSTRO = 0.03
MAX_TAMANO_ROSTRO = 0.85

# Umbrales de comparacion facial (embeddings AuraFace/glintr100)
UMBRAL_SIMILITUD = 0.60
# Confianza minima de deteccion (det_score de SCRFD) - mas estricta que el
# umbral interno del detector (0.5), para descartar detecciones marginales
MIN_DET_SCORE = 0.70
# Tamano maximo permitido por imagen (tamano real del archivo, no del base64)
MAX_IMAGEN_TAMANO_BYTES = 5 * 1024 * 1024  # 5 MB

# Base64 infla el tamano original en ~33% (codifica cada 3 bytes en 4 caracteres)
MAX_IMAGEN_BASE64_CHARS = int(MAX_IMAGEN_TAMANO_BYTES * 4 / 3)

# Angulo del rostro (heuristico con landmarks) - usados solo si se activa
# validar_angulo en validations.py. REVISAR Y CALIBRAR con fotos reales.
MAX_ROLL_GRADOS = 20        # inclinacion lateral maxima permitida
MAX_YAW_RATIO = 0.55        # que tan centrada debe estar la nariz respecto a los ojos