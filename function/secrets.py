import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv(override=True)


class Secrets:
    def __init__(self):
        object.__setattr__(self, "_env_vars", dict(os.environ))

    def __getattribute__(self, name: str) -> SecretStr:
        if name == "_env_vars":
            return object.__getattribute__(self, name)

        secret = os.getenv(name)
        if secret is None:
            raise ValueError(f"Secret '{name}' not found in environment variables")
        return SecretStr(secret)

    def __dir__(self) -> list[str]:
        return list(object.__getattribute__(self, "_env_vars").keys())


secrets = Secrets()
