# python -m uvicorn app.api:app --reload
# http://127.0.0.1:8000/docs
# python -m pip install -r requirements.txt
import asyncio
import datetime
import logging
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.config import settings
from app.core.face_engine import get_face_engine
from app.schemas import (VerificacionRequest, VerificacionResponse, ErrorResponse, CalidadImagen, TokenRequest, TokenResponse,)
from app.image_utils import base64_a_imagen
from app.security.jwt import verificar_token, verificar_credenciales_cliente, generar_token
from app.validations import (validar_resolucion, validar_un_rostro, validar_deteccion,validar_distancia, validar_iluminacion, agregar_error,)
from app.comparison import comparar_rostros_1a1
from app.enums import ResultadoVerificacion, TipoImagen, CodigoError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

logger = logging.getLogger(__name__)
app = FastAPI(title="API Identificacion Facial",description="API para verificacion facial 1:1",version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CARGA DEL MOTOR (SCRFD + AuraFace) - una sola vez al iniciar el proceso
face_engine = get_face_engine()
LADOS = (TipoImagen.ACTUAL, TipoImagen.REFERENCIA)

def _error_response(errores: list) -> dict:
    return {"exito": False, "errores": errores}

def _codigo_error_validacion(loc: tuple) -> CodigoError:
    """Traduce el 'loc' de un error de Pydantic al codigo de error propio de la API."""
    campo = loc[-1] if loc else None
    if campo in ("imagen_actual", "imagen_referencia"):
        return CodigoError.IMAGEN_INVALIDA
    return CodigoError.SOLICITUD_INVALIDA

def _traducir_mensaje(error: dict) -> str:
    """Traduce los mensajes que Pydantic genera en ingles para sus
    validaciones nativas (campo faltante, tipo invalido, largo invalido)."""
    tipo = error.get("type")
    campo = error["loc"][-1] if error.get("loc") else "el campo"

    if tipo == "missing":
        return f"El campo '{campo}' es obligatorio."
    if tipo == "string_type":
        return f"El campo '{campo}' debe ser un texto (string)."
    if tipo == "string_too_short":
        minimo = error.get("ctx", {}).get("min_length")
        return f"El campo '{campo}' no puede estar vacio (minimo {minimo} caracter)."
    if tipo == "string_too_long":
        maximo = error.get("ctx", {}).get("max_length")
        return f"El campo '{campo}' supera el tamano maximo permitido ({maximo} caracteres)."
    if tipo == "value_error":
        prefijo = "Value error, "
        msg = error["msg"]
        return msg[len(prefijo):] if msg.startswith(prefijo) else msg
    return error["msg"]

@app.exception_handler(RequestValidationError)
async def manejar_error_validacion(request: Request, exc: RequestValidationError):
    """Normaliza los errores 422 de Pydantic al mismo formato ErrorResponse"""
    errores = [
        {
            "codigo": _codigo_error_validacion(error["loc"]),
            "mensaje": _traducir_mensaje(error),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content=jsonable_encoder(_error_response(errores)))

@app.exception_handler(StarletteHTTPException)
async def manejar_http_exception(request: Request, exc: StarletteHTTPException):
    """Normaliza cualquier HTTPException (ej. 401 del JWT) al mismo formato
    ErrorResponse que usa el resto de la API."""
    detail = exc.detail
    if isinstance(detail, dict) and "codigo" in detail and "mensaje" in detail:
        errores = [detail]
    else:
        errores = [{"codigo": CodigoError.SOLICITUD_INVALIDA, "mensaje": str(detail)}]
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(_error_response(errores)))

@app.post("/auth/token", response_model=TokenResponse)
async def obtener_token(request: TokenRequest):
    if not verificar_credenciales_cliente(request.client_id, request.client_secret):
        raise HTTPException(
            status_code=401,
            detail={
                "codigo": CodigoError.CREDENCIALES_INVALIDAS,
                "mensaje": "client_id o client_secret invalidos.",
            },
        )

    expira_en_minutos = settings.access_token_expire_minutes
    token = generar_token(
        sub=request.client_id,
        expira_en=datetime.timedelta(minutes=expira_en_minutos),
    )
    return TokenResponse(access_token=token, expires_in=expira_en_minutos * 60)


@app.post("/reconocimiento", response_model=VerificacionResponse | ErrorResponse)
async def reconocimiento(request: VerificacionRequest, _token: dict = Depends(verificar_token)):
    errores = []
    imagenes_base64 = {TipoImagen.ACTUAL: request.imagen_actual,TipoImagen.REFERENCIA: request.imagen_referencia,
    }
    try:
        # CONVERTIR AMBAS IMAGENES DE BASE64 A OPENCV
        imagenes = {}
        for lado in LADOS:
            resultado = base64_a_imagen(imagenes_base64[lado], lado)
            agregar_error(errores, resultado)
            imagenes[lado] = resultado["imagen"] if resultado["valido"] else None
        if errores:
            return _error_response(errores)
        # VALIDAR RESOLUCION
        datos_calidad = {lado: {} for lado in LADOS}
        for lado in LADOS:
            resultado = validar_resolucion(imagenes[lado], lado)
            agregar_error(errores, resultado)
            if resultado["valido"]:
                datos_calidad[lado]["ancho"] = resultado["ancho"]
                datos_calidad[lado]["alto"] = resultado["alto"]
        if errores:
            return _error_response(errores)
        # DETECTAR ROSTROS Y CALCULAR EMBEDDINGS (bloqueante -> threadpool, en paralelo)
        resultados_deteccion = await asyncio.gather(
            *(run_in_threadpool(face_engine.get_faces, imagenes[lado]) for lado in LADOS)
        )
        rostros_detectados = dict(zip(LADOS, resultados_deteccion))
        # VALIDAR QUE HAYA EXACTAMENTE UN ROSTRO
        for lado in LADOS:
            resultado = validar_un_rostro(rostros_detectados[lado], lado)
            agregar_error(errores, resultado)
            if resultado["valido"]:
                datos_calidad[lado]["rostros_detectados"] = resultado["rostros_detectados"]
        if errores:
            return _error_response(errores)

        rostro = {lado: rostros_detectados[lado][0] for lado in LADOS}

        # VALIDAR CONFIANZA DE DETECCION, DISTANCIA E ILUMINACION
        for lado in LADOS:
            resultado = validar_deteccion(rostro[lado], lado)
            agregar_error(errores, resultado)
            if resultado["valido"]:
                datos_calidad[lado]["det_score"] = resultado["det_score"]
        for lado in LADOS:
            resultado = validar_distancia(imagenes[lado], rostro[lado], lado)
            agregar_error(errores, resultado)
            if resultado["valido"]:
                datos_calidad[lado]["distancia"] = resultado["distancia"]
        for lado in LADOS:
            resultado = validar_iluminacion(imagenes[lado], rostro[lado], lado)
            agregar_error(errores, resultado)
            if resultado["valido"]:
                datos_calidad[lado]["brillo"] = resultado["brillo"]
        if errores:
            return _error_response(errores)

        # COMPARACION FACIAL 1:1
        resultado_comparacion = comparar_rostros_1a1(rostro[TipoImagen.ACTUAL], rostro[TipoImagen.REFERENCIA])
        if not resultado_comparacion["valido"]:
            agregar_error(errores, resultado_comparacion)
            return _error_response(errores)

        resultado_verificacion = resultado_comparacion["resultado"]
        similitud = resultado_comparacion["similitud"]

        # ARMAR RESPUESTA (si llegamos aca, ambos lados pasaron todas las validaciones)
        calidad_actual = CalidadImagen(**datos_calidad[TipoImagen.ACTUAL])
        calidad_referencia = CalidadImagen(**datos_calidad[TipoImagen.REFERENCIA])

        return VerificacionResponse(exito=(resultado_verificacion == ResultadoVerificacion.COINCIDE),
            resultado=resultado_verificacion,
            similitud=similitud,
            imagen_actual=calidad_actual,
            imagen_referencia=calidad_referencia,
        )

    except Exception:
        logger.exception("Error inesperado procesando /reconocimiento")
        return _error_response([{
            "codigo": CodigoError.ERROR_INTERNO,
            "mensaje": "Ocurrio un error inesperado al procesar la solicitud."
        }])



        