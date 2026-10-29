"""Server-owned limits for durable job data, independent of paid-call budgets."""
import json

MIB = 1024 * 1024
RESPONSE_BYTES = 2 * MIB
JOB_RECEIPT_BYTES = 32 * MIB
# response and usage are stored separately. Reserve both at their maximum before
# dispatch, plus a conservative allowance for bounded identifiers/row metadata.
RECEIPT_METADATA_BYTES = 4096
RECEIPT_RESERVATION_BYTES = 2 * RESPONSE_BYTES + RECEIPT_METADATA_BYTES
JSON_LIMITS = {"response": RESPONSE_BYTES, "payload": 8 * MIB,
               "checkpoint": 16 * MIB, "result": 16 * MIB}


class StorageBudgetExceeded(ValueError):
    pass


def bounded_json_size(value, kind: str) -> int:
    """Measure Django's ASCII JSON representation, stopping at the ceiling.

    Limits include escaping expansion; a small UTF-8 response cannot expand into
    an unbounded persisted JSON string. No full duplicate document is allocated.
    """
    limit = JSON_LIMITS[kind]
    size = 0
    for part in json.JSONEncoder(ensure_ascii=True, allow_nan=False).iterencode(value):
        size += len(part)
        if size > limit:
            raise StorageBudgetExceeded(f"Job {kind} storage limit exceeded")
    return size
