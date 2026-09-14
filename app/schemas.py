import base64
import binascii
from pydantic import BaseModel, Field, field_validator
from app.enums import Angulo, CodigoError, Distancia, ResultadoVerificacion
from app.parameters import MAX_IMAGEN_BASE64_CHARS

class VerificacionRequest(BaseModel):
    """Datos que recibe la API para verificar dos imagenes faciales."""
    imagen_actual: str = Field(..., min_length=1, max_length=MAX_IMAGEN_BASE64_CHARS)
    imagen_referencia: str = Field(..., min_length=1, max_length=MAX_IMAGEN_BASE64_CHARS)

    @field_validator("imagen_actual", "imagen_referencia")
    @classmethod
    def validar_base64(cls, v: str) -> str:
        if v.startswith("data:"):
            try:
                v = v.split(",", 1)[1]
            except IndexError:
                raise ValueError("Formato data URI invalido, falta la parte base64")
        try:
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("La imagen no es un base64 valido")
        return v

class TokenRequest(BaseModel):
    """Credenciales que el backend manda para obtener un JWT de acceso."""
    client_id: str = Field(..., min_length=1)
    client_secret: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """JWT de acceso de corta duracion (formato estandar OAuth2)."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # segundos


class CalidadImagen(BaseModel):
    """Metricas de calidad evaluadas sobre una imagen individual."""
    ancho: int | None = Field(default=None, ge=0)
    alto: int | None = Field(default=None, ge=0)
    brillo: float | None = Field(default=None, ge=0)
    nitidez: float | None = None
    rostros_detectados: int | None = Field(default=None, ge=0)
    det_score: float | None = Field(default=None, ge=0, le=1)
    distancia: Distancia | None = None
    angulo: Angulo | None = None

class VerificacionResponse(BaseModel):
    """Respuesta exitosa de una verificacion facial."""
    exito: bool
    resultado: ResultadoVerificacion
    similitud: float = Field(ge=-1, le=1)
    imagen_actual: CalidadImagen
    imagen_referencia: CalidadImagen

class ErrorDetalle(BaseModel):
    """Un error individual dentro de la lista de errores de una respuesta fallida."""
    codigo: CodigoError
    mensaje: str

class ErrorResponse(BaseModel):
    """Respuesta cuando la verificacion no pudo completarse (uno o mas errores)."""
    exito: bool = False
    errores: list[ErrorDetalle]