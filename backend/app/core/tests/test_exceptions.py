from app.core.exceptions import (
    BusinessError,
    NotFoundError,
    ConflictError,
    DependencyError,
)


def test_business_error_has_fields():
    e = BusinessError("bad input", code="bad_input", context={"field": "qty"})
    assert e.detail == "bad input"
    assert e.code == "bad_input"
    assert e.context == {"field": "qty"}
    assert e.http_status == 400


def test_not_found_is_404():
    e = NotFoundError("nope", code="missing")
    assert e.http_status == 404


def test_conflict_is_409():
    e = ConflictError("stale", code="stale_data")
    assert e.http_status == 409


def test_dependency_is_503():
    e = DependencyError("baostock down", code="upstream")
    assert e.http_status == 503


def test_dict_payload():
    e = NotFoundError("x", code="m", context={"id": 1})
    assert e.as_response_dict() == {"detail": "x", "code": "m", "context": {"id": 1}}
