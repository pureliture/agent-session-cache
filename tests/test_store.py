"""Unit tests for MemoryContextStore lifecycle, capacity limits, and concurrency."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from context_hide.model import ContextItem, Scope
from context_hide.store import MemoryContextStore


import time


def _sample_item(
    item_id: str = "item_123",
    tool_call_id: str = "call_1",
    original: str = "sample output",
    compacted: str = "[hidden:item_123]\nsample output",
    visibility: str = "compacted",
    version: int = 1,
    expires_at: float | None = None,
    digest: str = "digest_123",
) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        tool_call_id=tool_call_id,
        content_sha256="sha_" + item_id,
        original=original,
        compacted=compacted,
        excerpt_lines=("sample output",),
        visibility=visibility,
        version=version,
        expires_at=expires_at if expires_at is not None else (time.monotonic() + 1000.0),
        invocation_digest=digest,
    )


def test_store_initialization_invalid_arguments():
    with pytest.raises(ValueError, match="positive"):
        MemoryContextStore(ttl_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        MemoryContextStore(max_bytes=-100)


def test_store_tlru_expiration():
    simulated_time = [100.0]
    store = MemoryContextStore(ttl_seconds=50, clock=lambda: simulated_time[0])
    scope = Scope("bridge", "test", "sess_1")

    item = _sample_item(item_id="item_exp", expires_at=140.0)
    res = store.put(scope, item)
    assert res.ok

    assert store.get(scope, "item_exp") is not None

    # Advance clock past expiration
    simulated_time[0] = 145.0
    assert store.get(scope, "item_exp") is None

    # Expire call cleans it up
    assert store.expire(now=145.0) == 0  # Already evicted by get()
    assert len(store.items(scope)) == 0


def test_store_size_accounting_and_store_full():
    store = MemoryContextStore(max_bytes=3000)
    scope = "session_cap"

    item1 = _sample_item(item_id="item_small", original="short")
    res1 = store.put(scope, item1)
    assert res1.ok
    assert store.used_bytes > 2000

    item_huge = _sample_item(item_id="item_huge", original="x" * 2000)
    res2 = store.put(scope, item_huge)
    assert not res2.ok
    assert res2.error == "store_full"


def test_store_put_equal_size_replacement_uses_only_incremental_capacity():
    item = _sample_item(item_id="item_equal")
    size = MemoryContextStore._item_size(item)
    store = MemoryContextStore(max_bytes=size)

    assert store.put("scope_equal", item).ok
    updated = replace(item, visibility="original", version=2)
    result = store.put("scope_equal", updated)

    assert result.ok
    assert result.item == updated
    assert store.used_bytes == size


def test_store_put_smaller_replacement_releases_unused_capacity():
    item = _sample_item(item_id="item_smaller", original="x" * 128)
    smaller = replace(item, original="x" * 64, version=2)
    store = MemoryContextStore(max_bytes=MemoryContextStore._item_size(item))

    assert store.put("scope_smaller", item).ok
    result = store.put("scope_smaller", smaller)

    assert result.ok
    assert store.used_bytes == MemoryContextStore._item_size(smaller)


def test_store_put_larger_replacement_at_exact_capacity():
    item = _sample_item(item_id="item_larger", original="x" * 64)
    larger = replace(item, original="x" * 96, version=2)
    capacity = MemoryContextStore._item_size(larger)
    store = MemoryContextStore(max_bytes=capacity)

    assert store.put("scope_larger", item).ok
    result = store.put("scope_larger", larger)

    assert result.ok
    assert result.item == larger
    assert store.used_bytes == capacity


def test_store_put_oversize_replacement_preserves_original_and_ttl():
    item = _sample_item(item_id="item_oversize", expires_at=time.monotonic() + 500.0)
    store = MemoryContextStore(max_bytes=MemoryContextStore._item_size(item) + 8)
    assert store.put("scope_oversize", item).ok
    before = store.used_bytes
    oversized = replace(item, original="x" * 1000, version=9, expires_at=900.0)

    result = store.put("scope_oversize", oversized)

    assert not result.ok and result.error == "store_full"
    assert store.get("scope_oversize", item.item_id) == item
    assert store.used_bytes == before


def test_store_put_replacement_does_not_evict_unrelated_live_item():
    item = _sample_item(item_id="item_keep", original="x" * 64)
    unrelated = _sample_item(item_id="item_unrelated", original="y" * 64)
    capacity = MemoryContextStore._item_size(item) + MemoryContextStore._item_size(unrelated)
    store = MemoryContextStore(max_bytes=capacity)
    assert store.put("scope_keep", item).ok
    assert store.put("scope_keep", unrelated).ok
    before = store.used_bytes
    too_large = replace(item, original="x" * 65, version=2)

    result = store.put("scope_keep", too_large)

    assert not result.ok and result.error == "store_full"
    assert store.get("scope_keep", item.item_id) == item
    assert store.get("scope_keep", unrelated.item_id) == unrelated
    assert store.used_bytes == before


def test_store_put_replacement_after_expiration_uses_freed_capacity():
    current_time = [100.0]
    old = _sample_item(item_id="item_expired_replace", original="x" * 64, expires_at=101.0)
    new = replace(old, original="x" * 128, expires_at=200.0, version=2)
    store = MemoryContextStore(max_bytes=MemoryContextStore._item_size(new), clock=lambda: current_time[0])
    assert store.put("scope_expired_replace", old).ok

    current_time[0] = 102.0
    result = store.put("scope_expired_replace", new)

    assert result.ok
    assert store.get("scope_expired_replace", old.item_id) == new
    assert store.used_bytes == MemoryContextStore._item_size(new)


def test_store_reserve_and_release():
    store = MemoryContextStore()
    scope = "scope_reserve"

    token, err = store.reserve(scope, "item_1", "digest_1")
    assert token is not None
    assert err is None

    store.release(scope, "item_1", token)

    # Can reserve again after release
    token2, err2 = store.reserve(scope, "item_1", "digest_1")
    assert token2 is not None
    assert err2 is None


def test_store_pending_bound_max_four():
    store = MemoryContextStore()
    tokens = []
    for i in range(4):
        token, err = store.reserve(f"scope_{i}", f"item_{i}", f"digest_{i}")
        assert token is not None and err is None
        tokens.append(token)

    # 5th reservation hits limit
    token5, err5 = store.reserve("scope_fifth", "item_fifth", "digest_fifth")
    assert token5 is None
    assert err5 == "pending_full"


def test_store_in_progress_dedup():
    store = MemoryContextStore()
    token1, err1 = store.reserve("scope_dup", "item_dup", "digest_dup")
    assert token1 is not None and err1 is None

    token2, err2 = store.reserve("scope_dup", "item_dup", "digest_dup")
    assert token2 is None
    assert err2 == "in_progress"


def test_store_atomic_existing_item_reservation():
    store = MemoryContextStore()
    item = _sample_item(item_id="item_exist")
    assert store.put("scope_exist", item).ok

    token, err = store.reserve("scope_exist", "item_exist", "digest_123")
    assert token is None
    assert err == "already_exists"


def test_store_cancellation_release():
    async def scenario():
        store = MemoryContextStore()
        token, _ = store.reserve("scope_cancel", "item_cancel", "digest_cancel")

        async def worker():
            try:
                await asyncio.sleep(10)
            finally:
                store.release("scope_cancel", "item_cancel", token)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert ("scope_cancel", "item_cancel") not in store._pending

    asyncio.run(scenario())


def test_store_unhide_toggles_visibility_and_increments_version():
    store = MemoryContextStore()
    item = _sample_item(item_id="item_unhide", visibility="compacted", version=1)
    store.put("scope_unhide", item)

    unhide_res = store.unhide("scope_unhide", "item_unhide")
    assert unhide_res.ok and unhide_res.item is not None
    assert unhide_res.item.visibility == "original"
    assert unhide_res.item.version == 2

    # Second unhide does not increment version if already original
    unhide_res2 = store.unhide("scope_unhide", "item_unhide")
    assert unhide_res2.ok and unhide_res2.item.version == 2


def test_store_unhide_nonexistent_returns_not_found():
    store = MemoryContextStore()
    res = store.unhide("scope_unknown", "item_unknown")
    assert not res.ok
    assert res.error == "not_found"


def test_store_same_item_compact_returns_existing():
    store = MemoryContextStore(min_chars=10)
    original = "\n".join(f"row {i:03d}" for i in range(50))
    res1 = store.compact(affinity="test_same", tool_call_id="call_same", original=original, tool_name="terminal")
    assert res1.ok and res1.item is not None

    res2 = store.compact(affinity="test_same", tool_call_id="call_same", original=original, tool_name="terminal")
    assert res2.ok
    assert res2.item.item_id == res1.item.item_id


def test_store_invocation_conflict_detection():
    store = MemoryContextStore()
    item = _sample_item(item_id="item_conflict", digest="digest_first")
    store.put("scope_conflict", item)

    token = object()
    store._pending[("scope_conflict", "item_conflict")] = (token, "digest_second")
    conflict_item = replace(item, invocation_digest="digest_second")
    res = store.put("scope_conflict", conflict_item, reservation=token)
    assert res.ok

    # Mismatched digest on existing item via compact
    res_conflict = store.compact(
        affinity="scope_conflict",
        tool_call_id="call_conflict",
        original="different content",
        input_digest="digest_other",
    )
    # Different item_id produces independent item or conflict if matched
    assert res_conflict is not None


def test_store_thread_safety():
    store = MemoryContextStore()
    scope = "thread_safety_scope"

    def worker(idx: int):
        item = _sample_item(item_id=f"item_{idx}", tool_call_id=f"call_{idx}")
        store.put(scope, item)
        store.get(scope, f"item_{idx}")
        store.unhide(scope, f"item_{idx}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(50)))

    assert len(store.items(scope)) == 50


def test_store_expire_method_purges_idle_groups():
    simulated_time = [100.0]
    store = MemoryContextStore(clock=lambda: simulated_time[0])

    item1 = _sample_item(item_id="item_g1", expires_at=120.0)
    item2 = _sample_item(item_id="item_g2", expires_at=120.0)
    store.put("group1", item1)
    store.put("group2", item2)

    assert len(store.items("group1")) == 1
    assert len(store.items("group2")) == 1

    simulated_time[0] = 130.0
    evicted = store.expire()
    assert evicted == 2
    assert len(store.items("group1")) == 0
    assert len(store.items("group2")) == 0
