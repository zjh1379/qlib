from typing import Any


class BusinessError(Exception):
    """Base for all expected business-logic errors. Maps to HTTP 400 by default."""

    http_status: int = 400

    def __init__(self, detail: str, code: str, context: dict[str, Any] | None = None):
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.context = context or {}

    def as_response_dict(self) -> dict[str, Any]:
        return {"detail": self.detail, "code": self.code, "context": self.context}


class NotFoundError(BusinessError):
    http_status = 404


class ConflictError(BusinessError):
    http_status = 409


class DependencyError(BusinessError):
    """External dependency (qlib, baostock, file system) failed."""
    http_status = 503
