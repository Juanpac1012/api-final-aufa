from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
APP_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = APP_DIR.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"),env_file_encoding="utf-8",extra="ignore",)
    # --- Modelos (SCRFD + AuraFace, descargados de fal/AuraFace-v1) ---
    models_dir: Path = APP_DIR / "models"
    detector_filename: str = "scrfd_10g_bnkps.onnx"
    recognizer_filename: str = "glintr100.onnx"
    # --- Inferencia ONNX Runtime ---
    onnx_providers: list[str] = ["CPUExecutionProvider"]
    detection_size: tuple[int, int] = (640, 640)
    detection_threshold: float = 0.5
    # --- CORS ---
    # "*" = cualquier origen puede llamar la API desde el navegador.
    # Cuando exista un frontend real, restringir aca (o via .env), ej:
    # CORS_ALLOW_ORIGINS=["https://miapp.com"]
    cors_allow_origins: list[str] = ["*"]
    # --- JWT (HS256) ---
    # Sin default a proposito: si no esta en .env, la app falla al arrancar
    # en vez de correr con un secreto inseguro conocido.
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    # --- Autenticacion cliente (POST /auth/token) ---
    # Credenciales del backend que consume esta API. El backend las manda a
    # /auth/token y recibe a cambio un JWT de corta duracion.
    auth_client_id: str
    auth_client_secret: str
    access_token_expire_minutes: int = 30

    @property
    def detector_path(self) -> str:
        return str(self.models_dir / self.detector_filename)

    @property
    def recognizer_path(self) -> str:
        return str(self.models_dir / self.recognizer_filename)

    @property
    def ctx_id(self) -> int:
        return 0 if "CUDAExecutionProvider" in self.onnx_providers else -1

settings = Settings()
