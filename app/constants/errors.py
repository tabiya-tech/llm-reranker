from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """The body of every error response the API returns (4xx and 5xx, including 422)."""

    detail: str = Field(description="What went wrong.")


def error_response_doc(description: str, detail: str) -> dict[str, Any]:
    """OpenAPI ``responses`` entry for an error status: ``ErrorResponse`` plus an example."""
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": detail}}},
    }
