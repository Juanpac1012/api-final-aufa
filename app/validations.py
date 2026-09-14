import cv2
import numpy as np
from app.enums import CodigoError, Distancia, TipoImagen
from app.parameters import (MIN_ANCHO, MIN_ALTO,MIN_BRILLO, MAX_BRILLO,MIN_ANCHO_ROSTRO,MIN_DET_SCORE,MIN_TAMANO_ROSTRO, MAX_TAMANO_ROSTRO,)

def obtener_codigo_error(codigo_actual: CodigoError, codigo_referencia: CodigoError,tipo_imagen: TipoImagen) -> CodigoError:
    if tipo_imagen == TipoImagen.ACTUAL:
        return codigo_actual
    return codigo_referencia

def agregar_error(errores: list, validacion: dict) -> None:
    if not validacion["valido"]:
        errores.append({
            "codigo": validacion["codigo"],
            "mensaje": validacion["mensaje"]
        })

def validar_resolucion(imagen, tipo_imagen: TipoImagen) -> dict:
    alto, ancho = imagen.shape[:2]
    if ancho < MIN_ANCHO or alto < MIN_ALTO:
        codigo = obtener_codigo_error(
            CodigoError.RESOLUCION_INVALIDA_ACTUAL,
            CodigoError.RESOLUCION_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"La resolucion de la imagen {tipo_imagen.value} es insuficiente. "
                f"Resolucion recibida: {ancho}x{alto}. "
                f"Resolucion minima requerida: {MIN_ANCHO}x{MIN_ALTO}."
            )
        }
    return {"valido": True, "ancho": int(ancho), "alto": int(alto)}

def validar_un_rostro(rostros, tipo_imagen: TipoImagen) -> dict:
    cantidad_rostros = len(rostros)
    if cantidad_rostros == 0:
        codigo = obtener_codigo_error(
            CodigoError.ROSTRO_ACTUAL_NO_DETECTADO,
            CodigoError.ROSTRO_REFERENCIA_NO_DETECTADO,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": f"No se detecto ningun rostro en la imagen {tipo_imagen.value}."
        }

    if cantidad_rostros > 1:
        codigo = obtener_codigo_error(
            CodigoError.MULTIPLES_ROSTROS_ACTUAL,
            CodigoError.MULTIPLES_ROSTROS_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"Se detectaron {cantidad_rostros} rostros en la imagen {tipo_imagen.value}. "
                f"Debe existir exactamente un rostro."
            )
        }
    return {"valido": True, "rostros_detectados": cantidad_rostros}


def validar_deteccion(rostro, tipo_imagen: TipoImagen) -> dict:
    """Rechaza detecciones marginales: SCRFD puede reportar un rostro con
    confianza apenas por encima de su propio umbral interno."""
    det_score = float(rostro.det_score)
    if det_score < MIN_DET_SCORE:
        codigo = obtener_codigo_error(
            CodigoError.DETECCION_BAJA_CONFIANZA_ACTUAL,
            CodigoError.DETECCION_BAJA_CONFIANZA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"La deteccion del rostro en la imagen {tipo_imagen.value} tiene "
                f"baja confianza ({det_score:.2f}). Minimo requerido: {MIN_DET_SCORE}."
            )
        }

    return {"valido": True, "det_score": det_score}

def validar_distancia(imagen, rostro, tipo_imagen: TipoImagen) -> dict:
    x1, y1, x2, y2 = rostro.bbox
    ancho_rostro = float(x2 - x1)
    alto_rostro = float(y2 - y1)
    if ancho_rostro < MIN_ANCHO_ROSTRO:
        codigo = obtener_codigo_error(
            CodigoError.DISTANCIA_INVALIDA_ACTUAL,
            CodigoError.DISTANCIA_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"El rostro de la imagen {tipo_imagen.value} esta demasiado lejos "
                f"o es demasiado pequeno. Ancho detectado: {ancho_rostro:.0f}px. "
                f"Minimo requerido: {MIN_ANCHO_ROSTRO}px."
            )
        }

    # PROPORCION DEL ROSTRO RESPECTO A LA IMAGEN COMPLETA
    alto_imagen, ancho_imagen = imagen.shape[:2]
    area_imagen = float(ancho_imagen * alto_imagen)
    proporcion = (ancho_rostro * alto_rostro) / area_imagen if area_imagen > 0 else 0.0

    if proporcion > MAX_TAMANO_ROSTRO:
        codigo = obtener_codigo_error(
            CodigoError.DISTANCIA_INVALIDA_ACTUAL,
            CodigoError.DISTANCIA_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"El rostro de la imagen {tipo_imagen.value} esta demasiado cerca de la camara. "
                f"Ocupa {proporcion:.0%} de la imagen. Maximo permitido: {MAX_TAMANO_ROSTRO:.0%}."
            ),
            "distancia": Distancia.CERCA,
        }

    if proporcion < MIN_TAMANO_ROSTRO:
        codigo = obtener_codigo_error(
            CodigoError.DISTANCIA_INVALIDA_ACTUAL,
            CodigoError.DISTANCIA_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"El rostro de la imagen {tipo_imagen.value} esta demasiado lejos de la camara. "
                f"Ocupa {proporcion:.0%} de la imagen. Minimo requerido: {MIN_TAMANO_ROSTRO:.0%}."
            ),
            "distancia": Distancia.LEJOS,
        }

    return {
        "valido": True,
        "ancho_rostro": ancho_rostro,
        "alto_rostro": alto_rostro,
        "distancia": Distancia.CORRECTA,
    }

def obtener_recorte_rostro(imagen, rostro):
    x1, y1, x2, y2 = map(int, rostro.bbox)
    alto, ancho = imagen.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(ancho, x2)
    y2 = min(alto, y2)

    if x1 >= x2 or y1 >= y2:
        return None
    rostro_recortado = imagen[y1:y2, x1:x2]
    if rostro_recortado.size == 0:
        return None
    return rostro_recortado

def validar_iluminacion(imagen, rostro, tipo_imagen: TipoImagen) -> dict:
    rostro_recortado = obtener_recorte_rostro(imagen, rostro)
    if rostro_recortado is None:
        codigo = obtener_codigo_error(
            CodigoError.ILUMINACION_INVALIDA_ACTUAL,
            CodigoError.ILUMINACION_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"No se pudo obtener correctamente el rostro de la imagen "
                f"{tipo_imagen.value} para analizar la iluminacion."
            )
        }
    gris = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2GRAY)
    brillo_promedio = float(np.mean(gris))
    if brillo_promedio < MIN_BRILLO:
        codigo = obtener_codigo_error(
            CodigoError.ILUMINACION_INVALIDA_ACTUAL,
            CodigoError.ILUMINACION_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"El rostro de la imagen {tipo_imagen.value} esta demasiado oscuro. "
                f"Brillo detectado: {brillo_promedio:.2f}. Minimo permitido: {MIN_BRILLO}."
            )
        }
    if brillo_promedio > MAX_BRILLO:
        codigo = obtener_codigo_error(
            CodigoError.ILUMINACION_INVALIDA_ACTUAL,
            CodigoError.ILUMINACION_INVALIDA_REFERENCIA,
            tipo_imagen
        )
        return {
            "valido": False,
            "codigo": codigo,
            "mensaje": (
                f"El rostro de la imagen {tipo_imagen.value} esta demasiado iluminado. "
                f"Brillo detectado: {brillo_promedio:.2f}. Maximo permitido: {MAX_BRILLO}."
            )
        }
    return {"valido": True, "brillo": brillo_promedio}


# PENDIENTE DE ACTIVAR - descomentar cuando se decida usarlas.
# Requieren importar arriba: import math, y de app.enums: Angulo
# y de app.parameters: MIN_NITIDEZ, MAX_ROLL_GRADOS, MAX_YAW_RATIO

# def validar_nitidez(rostro_recortado, tipo_imagen: TipoImagen) -> dict:
#     """Detecta desenfoque via la varianza del Laplaciano: mientras mas baja,
#     mas borroso esta el rostro. Un rostro borroso produce un embedding menos
#     discriminativo y es una causa comun de falsos rechazos/aceptaciones."""
#     gris = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2GRAY)
#     nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
#
#     if nitidez < MIN_NITIDEZ:
#         codigo = obtener_codigo_error(
#             CodigoError.NITIDEZ_INVALIDA_ACTUAL,
#             CodigoError.NITIDEZ_INVALIDA_REFERENCIA,
#             tipo_imagen
#         )
#         return {
#             "valido": False,
#             "codigo": codigo,
#             "mensaje": (
#                 f"El rostro de la imagen {tipo_imagen.value} esta demasiado borroso. "
#                 f"Nitidez detectada: {nitidez:.1f}. Minimo requerido: {MIN_NITIDEZ}."
#             )
#         }
#
#     return {"valido": True, "nitidez": nitidez}


# def validar_angulo(rostro, tipo_imagen: TipoImagen) -> dict:
#     """Estima la pose del rostro a partir de los 5 landmarks que entrega
#     SCRFD (ojo_izq, ojo_der, nariz, boca_izq, boca_der).
#
#     NOTA: es un heuristico geometrico simple, no un estimador de pose 3D.
#     Los umbrales (MAX_ROLL_GRADOS, MAX_YAW_RATIO) son un punto de partida
#     y deben calibrarse con fotos reales antes de confiar en ellos.
#     """
#     if rostro.kps is None or len(rostro.kps) < 5:
#         return {"valido": True, "angulo": None}
#
#     ojo_izq, ojo_der, nariz, boca_izq, boca_der = rostro.kps[:5]
#
#     # ROLL: inclinacion lateral de la cabeza (angulo de la linea entre ojos)
#     dy = float(ojo_der[1] - ojo_izq[1])
#     dx = float(ojo_der[0] - ojo_izq[0])
#     roll_grados = abs(math.degrees(math.atan2(dy, dx)))
#
#     if roll_grados > MAX_ROLL_GRADOS:
#         codigo = obtener_codigo_error(
#             CodigoError.ANGULO_INVALIDO_ACTUAL,
#             CodigoError.ANGULO_INVALIDO_REFERENCIA,
#             tipo_imagen
#         )
#         return {
#             "valido": False,
#             "codigo": codigo,
#             "mensaje": (
#                 f"El rostro de la imagen {tipo_imagen.value} esta inclinado "
#                 f"({roll_grados:.1f} grados). Maximo permitido: {MAX_ROLL_GRADOS} grados."
#             ),
#             "angulo": Angulo.INCLINADO,
#         }
#
#     # YAW: giro izquierda/derecha, segun que tan centrada esta la nariz
#     # respecto a los ojos (normalizado por la distancia entre ojos)
#     distancia_ojos = math.hypot(dx, dy)
#     if distancia_ojos == 0:
#         return {"valido": True, "angulo": Angulo.FRONTAL}
#
#     centro_ojos_x = (ojo_izq[0] + ojo_der[0]) / 2
#     yaw_ratio = abs(float(nariz[0]) - centro_ojos_x) / distancia_ojos
#
#     if yaw_ratio > MAX_YAW_RATIO:
#         codigo = obtener_codigo_error(
#             CodigoError.ANGULO_INVALIDO_ACTUAL,
#             CodigoError.ANGULO_INVALIDO_REFERENCIA,
#             tipo_imagen
#         )
#         return {
#             "valido": False,
#             "codigo": codigo,
#             "mensaje": (
#                 f"El rostro de la imagen {tipo_imagen.value} no esta de frente. "
#                 f"Debe mirar hacia la camara."
#             ),
#             "angulo": Angulo.PERFIL,
#         }
#
#     return {"valido": True, "angulo": Angulo.FRONTAL}