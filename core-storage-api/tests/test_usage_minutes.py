"""Usage metering's storage half (migration 066), against a live PostgreSQL.

What a flush must guarantee: counted once even when re-sent, added (not
replaced) across flushes and instances, dropped for a tenant that switched
metering off, and readable per tenant grouped by day, agent or operation.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

PREFIX = "/api/v1/storage"
MINUTE = "2026-10-11T12:30:00+00:00"
NEXT_DAY = "2026-10-12T08:00:00+00:00"


def _llm(tenant: str, **over) -> dict:
    row = {
        "minute": MINUTE,
        "tenant_id": tenant,
        "agent_id": "a1",
        "fleet_id": "",
        "operation": "enrichment",
        "kind": "llm",
        "provider": "openai",
        "model": "gpt-test",
        "calls": 2,
        "failures": 1,
        "tokens_in": 100,
        "tokens_out": 20,
        "tokens_reasoning": 5,
    }
    row.update(over)
    return row


def _value(tenant: str, metric: str, quantity: int, **over) -> dict:
    row = {
        "minute": MINUTE,
        "tenant_id": tenant,
        "agent_id": "a1",
        "fleet_id": "",
        "operation": "search",
        "metric": metric,
        "quantity": quantity,
    }
    row.update(over)
    return row


async def _flush(client: AsyncClient, *, batch_id: str | None = None, llm=(), value=()) -> dict:
    r = await client.post(
        f"{PREFIX}/usage/flush",
        json={"batch_id": batch_id or str(uuid.uuid4()), "llm": list(llm), "value": list(value)},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _query(client: AsyncClient, tenant: str, group_by: list[str]) -> dict:
    r = await client.post(
        f"{PREFIX}/usage/query",
        params={"tenant_id": tenant},
        json={"start": "2026-10-01T00:00:00+00:00", "end": "2026-11-01T00:00:00+00:00", "group_by": group_by},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def test_a_resent_batch_is_counted_once(client: AsyncClient):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    batch = str(uuid.uuid4())
    assert (await _flush(client, batch_id=batch, llm=[_llm(tenant)]))["applied"] is True
    assert (await _flush(client, batch_id=batch, llm=[_llm(tenant)]))["applied"] is False
    [row] = (await _query(client, tenant, ["operation"]))["model_calls"]
    assert (row["calls"], row["failures"], row["tokens_in"], row["tokens_out"], row["tokens_reasoning"]) == (
        2,
        1,
        100,
        20,
        5,
    )


async def test_two_flushes_of_one_minute_add_up(client: AsyncClient):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    await _flush(client, llm=[_llm(tenant)])
    await _flush(client, llm=[_llm(tenant, calls=3, tokens_in=50)])
    [row] = (await _query(client, tenant, []))["model_calls"]
    assert (row["calls"], row["tokens_in"]) == (5, 150)


async def test_the_read_groups_by_day_agent_and_operation(client: AsyncClient):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    await _flush(
        client,
        llm=[
            _llm(tenant),
            _llm(tenant, agent_id="a2"),
            _llm(tenant, minute=NEXT_DAY),
            _llm(tenant, operation="embed", kind="embedding"),
        ],
    )
    by_day = {r["day"]: r["calls"] for r in (await _query(client, tenant, ["day"]))["model_calls"]}
    assert by_day == {"2026-10-11": 6, "2026-10-12": 2}
    by_agent = {
        r["agent_id"]: r["calls"] for r in (await _query(client, tenant, ["agent_id"]))["model_calls"]
    }
    assert by_agent == {"a1": 6, "a2": 2}
    by_op = {r["operation"]: r["calls"] for r in (await _query(client, tenant, ["operation"]))["model_calls"]}
    assert by_op == {"enrichment": 6, "embed": 2}


async def test_savings_are_always_broken_out_by_metric(client: AsyncClient):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    await _flush(
        client,
        value=[
            _value(tenant, "recall.source_tokens", 900),
            _value(tenant, "recall.returned_tokens", 120),
            _value(tenant, "recall.source_tokens", 100, agent_id="a2"),
        ],
    )
    got = {r["metric"]: r["quantity"] for r in (await _query(client, tenant, []))["savings"]}
    assert got == {"recall.source_tokens": 1000, "recall.returned_tokens": 120}


async def test_a_tenant_that_switched_metering_off_records_nothing(client: AsyncClient):
    off, on = f"t-off-{uuid.uuid4().hex[:8]}", f"t-on-{uuid.uuid4().hex[:8]}"
    r = await client.post(
        f"{PREFIX}/organization-settings/{off}",
        json={"settings": {"observability": {"usage_metering_enabled": False}}},
    )
    assert r.status_code in (200, 201), r.text
    await _flush(client, llm=[_llm(off), _llm(on)], value=[_value(off, "recall.source_tokens", 5)])
    assert (await _query(client, off, []))["model_calls"] == []
    assert (await _query(client, off, []))["savings"] == []
    assert len((await _query(client, on, []))["model_calls"]) == 1


async def test_the_read_is_bound_to_one_tenant(client: AsyncClient):
    tenant, other = f"t-{uuid.uuid4().hex[:8]}", f"t-{uuid.uuid4().hex[:8]}"
    await _flush(client, llm=[_llm(tenant), _llm(other)])
    assert len((await _query(client, tenant, []))["model_calls"]) == 1
    r = await client.post(f"{PREFIX}/usage/query", json={"start": MINUTE, "end": NEXT_DAY})
    assert r.status_code == 422


async def test_a_row_without_a_tenant_or_with_a_negative_count_is_refused(client: AsyncClient):
    bad_tenant = {k: v for k, v in _llm("x").items() if k != "tenant_id"}
    r = await client.post(f"{PREFIX}/usage/flush", json={"batch_id": str(uuid.uuid4()), "llm": [bad_tenant]})
    assert r.status_code == 422
    r = await client.post(
        f"{PREFIX}/usage/flush", json={"batch_id": str(uuid.uuid4()), "llm": [_llm("x", calls=-1)]}
    )
    assert r.status_code == 422
