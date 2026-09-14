import datetime
import hmac

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.enums import CodigoError

# auto_error=False: si falta el header Authorization, no queremos que
# HTTPBearer levante su propio error generico ("Not authenticated") - se
# maneja aca abajo para que el mensaje/codigo sea consistente con el resto
# de la API.
_security_scheme = HTTPBearer(auto_error=False)

def _error(status_code: int, codigo: CodigoError, mensaje: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"codigo": codigo, "mensaje": mensaje})

def verificar_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security_scheme),
) -> dict:
    """Dependencia de FastAPI: exige un JWT valido (HS256) en el header
    'Authorization: Bearer <token>'. Se agrega a un endpoint con
    Depends(verificar_token)."""
    if credentials is None:
        raise _error(401, CodigoError.TOKEN_FALTANTE, "Falta el token de autenticacion.")

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise _error(401, CodigoError.TOKEN_EXPIRADO, "El token de autenticacion expiro.")
    except jwt.InvalidTokenError:
        raise _error(401, CodigoError.TOKEN_INVALIDO, "El token de autenticacion no es valido.")

    return payload

def generar_token(sub: str, expira_en: datetime.timedelta) -> str:
    """Genera un JWT (HS256) firmado con el secreto de la app."""
    ahora = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": sub,
        "iat": ahora,
        "exp": ahora + expira_en,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

def verificar_credenciales_cliente(client_id: str, client_secret: str) -> bool:
    """Compara client_id/client_secret contra los configurados en .env.
    Usa comparacion de tiempo constante (hmac.compare_digest) para no dar
    pistas de timing sobre en que caracter difiere un secreto incorrecto."""
    id_ok = hmac.compare_digest(client_id, settings.auth_client_id)
    secret_ok = hmac.compare_digest(client_secret, settings.auth_client_secret)
    return id_ok and secret_ok


if __name__ == "__main__":
    # Uso manual/dev: python -m app.security.jwt [--dias 30] [--sub "cliente-interno"]
    # El flujo "real" para el backend es POST /auth/token (ver README).
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Genera un token JWT para la API (uso manual/dev)")
    parser.add_argument("--dias", type=int, default=30, help="Dias de validez del token (default: 30)")
    parser.add_argument("--sub", type=str, default="cliente-interno", help="Identificador del cliente/servicio")
    args = parser.parse_args()

    token = generar_token(sub=args.sub, expira_en=datetime.timedelta(days=args.dias))
    print(token)
    print(f"\nValido {args.dias} dias, para '{args.sub}'.", file=sys.stderr)
