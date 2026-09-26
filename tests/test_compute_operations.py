from __future__ import annotations

from datetime import UTC, datetime

from app.compute.service import ComputeOperationsService
from app.core.clock import FrozenClock
from app.database import get_connection, transaction


TEMPLATE = {
    "code": "solver-a",
    "name": "方程求解模板",
    "algorithm": "solver-a",
    "parameter_schema": {
        "iterations": {"type": "integer", "required": True, "minimum": 1, "maximum": 10000},
        "tolerance": {"type": "number", "required": False, "minimum": 0.0, "maximum": 1.0},
        "mode": {"type": "string", "required": True, "choices": ["fast", "accurate"]},
    },
    "default_parameters": {"tolerance": 0.001},
    "max_runtime_seconds": 300,
    "max_attempts": 2,
}


def submit_payload(key: str, *, user: str = "researcher-1", priority: int = 50) -> dict:
    return {
        "template_code": "solver-a",
        "project_code": "project-a",
        "requested_by": user,
        "parameters": {"iterations": 100, "mode": "accurate"},
        "priority": priority,
        "idempotency_key": key,
    }


def create_template(client) -> None:
    response = client.post("/api/compute/templates?actor=administrator", json=TEMPLATE)
    assert response.status_code == 201, response.text


def test_template_submission_idempotency_and_parameter_validation(client):
    create_template(client)
    first = client.post("/api/compute/tasks", json=submit_payload("request-000001"))
    second = client.post("/api/compute/tasks", json=submit_payload("request-000001"))
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    invalid = submit_payload("request-000002")
    invalid["parameters"]["iterations"] = 20000
    rejected = client.post("/api/compute/tasks", json=invalid)
    assert rejected.status_code == 422


def test_priority_capability_claim_and_result_version(client):
    create_template(client)
    low = client.post("/api/compute/tasks", json=submit_payload("priority-low", priority=10)).json()
    high = client.post("/api/compute/tasks", json=submit_payload("priority-high", priority=90)).json()
    no_match = client.post("/api/compute/tasks/claim", json={"worker_id": "w0", "capabilities": ["other"], "lease_seconds": 60})
    assert no_match.status_code == 200 and no_match.json()["task"] is None
    claimed = client.post("/api/compute/tasks/claim", json={"worker_id": "w1", "capabilities": ["solver-a"], "lease_seconds": 60})
    assert claimed.status_code == 200
    assert claimed.json()["task"]["id"] == high["id"]
    completed = client.post(
        f"/api/compute/tasks/{high['id']}/complete",
        json={"worker_id": "w1", "result": {"value": 3.14}, "metrics": {"seconds": 2}},
    )
    assert completed.status_code == 200
    details = client.get(f"/api/compute/task-details/{high['id']}").json()
    assert details["status"] == "succeeded"
    assert details["current_result_version"] == 1
    assert len(details["results"]) == 1
    assert low["status"] == "queued"


def test_quota_cancel_retry_priority_and_batch_interventions(client):
    create_template(client)
    quota = client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "limited", "max_queued": 1, "max_running": 1, "daily_submissions": 2},
    )
    assert quota.status_code == 200
    one = client.post("/api/compute/tasks", json=submit_payload("quota-one", user="limited")).json()
    blocked = client.post("/api/compute/tasks", json=submit_payload("quota-two", user="limited"))
    assert blocked.status_code == 409
    cancelled = client.post(f"/api/compute/tasks/{one['id']}/cancel", json={"actor": "administrator", "reason": "项目暂停"})
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    retried = client.post(f"/api/compute/tasks/{one['id']}/retry", json={"actor": "administrator", "reason": "项目恢复", "priority": 95})
    assert retried.status_code == 200 and retried.json()["priority"] == 95
    other = client.post("/api/compute/tasks", json=submit_payload("batch-other", user="other-user")).json()
    batch = client.post(
        "/api/compute/tasks/batch",
        json={"task_ids": [one["id"], other["id"]], "operation": "priority", "actor": "administrator", "reason": "紧急算例", "priority": 99},
    )
    assert batch.status_code == 200
    assert len(batch.json()["succeeded"]) == 2
    details = client.get(f"/api/compute/task-details/{one['id']}").json()
    assert [item["action"] for item in details["interventions"]] == ["cancel", "retry", "priority"]


def test_failure_backoff_and_expired_lease_recovery(client):
    from app.database import init_db

    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 2, 0, tzinfo=UTC))
    service = ComputeOperationsService(get_connection(), clock)
    service.create_template(TEMPLATE, "administrator")
    first = service.submit(submit_payload("failure-000001"))
    claimed = service.claim("worker-a", ["solver-a"], 10)
    assert claimed and claimed["id"] == first["id"]
    failed = service.fail(first["id"], "worker-a", "numeric_error", "数值不收敛", True)
    assert failed["status"] == "queued"
    assert failed["available_at"] > failed["updated_at"]
    clock.advance(seconds=2)
    claimed_again = service.claim("worker-a", ["solver-a"], 10)
    assert claimed_again and claimed_again["attempt_count"] == 2
    clock.advance(seconds=11)
    recovered = service.recover_expired()
    assert recovered["exhausted"] == [first["id"]]
    details = service.get_task(first["id"])
    assert details["status"] == "failed"
    assert details["interventions"][-1]["action"] == "lease_recovery"


def test_claim_rechecks_running_quota_and_skips_full_accounts(client):
    create_template(client)
    quota = client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "alice", "max_queued": 10, "max_running": 1, "daily_submissions": 20},
    )
    assert quota.status_code == 200
    first = client.post("/api/compute/tasks", json=submit_payload("alice-first", user="alice", priority=90)).json()
    second = client.post("/api/compute/tasks", json=submit_payload("alice-second", user="alice", priority=80)).json()
    other = client.post("/api/compute/tasks", json=submit_payload("bob-first", user="bob", priority=10)).json()

    def claim(worker: str) -> dict | None:
        return client.post("/api/compute/tasks/claim", json={"worker_id": worker, "capabilities": ["solver-a"], "lease_seconds": 60}).json()["task"]

    assert claim("w1")["id"] == first["id"]
    # alice 运行名额已满，队首的 alice 第二条任务不能阻塞 bob 的任务
    assert claim("w2")["id"] == other["id"]
    assert claim("w3") is None
    # 完成后名额立即释放，alice 的下一条任务可以被领取
    completed = client.post(f"/api/compute/tasks/{first['id']}/complete", json={"worker_id": "w1", "result": {"value": 1}, "metrics": {}})
    assert completed.status_code == 200
    assert claim("w4")["id"] == second["id"]


def test_running_slot_released_on_failure_cancel_and_lease_recovery(client):
    from app.database import init_db

    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 2, 0, tzinfo=UTC))
    service = ComputeOperationsService(get_connection(), clock)
    service.create_template(TEMPLATE, "administrator")
    service.set_quota(
        {"subject_type": "user", "subject_key": "researcher-1", "max_queued": 10, "max_running": 1, "daily_submissions": 20},
        "administrator",
    )
    first = service.submit(submit_payload("release-000001"))
    second = service.submit(submit_payload("release-000002"))
    third = service.submit(submit_payload("release-000003"))

    assert service.claim("worker-a", ["solver-a"], 10)["id"] == first["id"]
    assert service.claim("worker-b", ["solver-a"], 10) is None
    # 不可重试的失败立即释放名额
    service.fail(first["id"], "worker-a", "numeric_error", "数值不收敛", False)
    assert service.claim("worker-b", ["solver-a"], 10)["id"] == second["id"]
    assert service.claim("worker-c", ["solver-a"], 10) is None
    # 取消运行中的任务立即释放名额
    service.cancel(second["id"], "administrator", "项目暂停")
    assert service.claim("worker-c", ["solver-a"], 10)["id"] == third["id"]
    assert service.claim("worker-d", ["solver-a"], 10) is None
    # 租约过期恢复后立即释放名额
    clock.advance(seconds=11)
    assert service.recover_expired()["recovered"] == [third["id"]]
    assert service.claim("worker-d", ["solver-a"], 10)["id"] == third["id"]


def test_quota_adjustment_affects_future_claims_without_preemption(client):
    create_template(client)
    client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "bob", "max_queued": 10, "max_running": 2, "daily_submissions": 20},
    )
    first = client.post("/api/compute/tasks", json=submit_payload("adjust-000001", user="bob")).json()
    second = client.post("/api/compute/tasks", json=submit_payload("adjust-000002", user="bob")).json()
    third = client.post("/api/compute/tasks", json=submit_payload("adjust-000003", user="bob")).json()

    def claim(worker: str) -> dict | None:
        return client.post("/api/compute/tasks/claim", json={"worker_id": worker, "capabilities": ["solver-a"], "lease_seconds": 60}).json()["task"]

    assert claim("w1")["id"] == first["id"]
    assert claim("w2")["id"] == second["id"]
    # 收紧配额不强制中断已有运行，但后续领取被阻止
    client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "bob", "max_queued": 10, "max_running": 1, "daily_submissions": 20},
    )
    assert claim("w3") is None
    assert client.get(f"/api/compute/task-details/{first['id']}").json()["status"] == "running"
    assert client.get(f"/api/compute/task-details/{second['id']}").json()["status"] == "running"
    # 放宽配额后后续领取恢复
    client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "bob", "max_queued": 10, "max_running": 3, "daily_submissions": 20},
    )
    assert claim("w4")["id"] == third["id"]


def test_concurrent_claims_never_exceed_running_quota(client):
    import threading

    create_template(client)
    for user, limit in (("alice", 1), ("bob", 2)):
        quota = client.put(
            "/api/compute/quotas?actor=administrator",
            json={"subject_type": "user", "subject_key": user, "max_queued": 10, "max_running": limit, "daily_submissions": 20},
        )
        assert quota.status_code == 200
    alice_tasks = [client.post("/api/compute/tasks", json=submit_payload(f"alice-{index:06d}", user="alice", priority=90 - index)).json() for index in range(3)]
    bob_tasks = [client.post("/api/compute/tasks", json=submit_payload(f"bob-{index:06d}", user="bob", priority=60 - index)).json() for index in range(3)]

    worker_count = 6
    barrier = threading.Barrier(worker_count)
    lock = threading.Lock()
    claimed: list[dict] = []
    errors: list[Exception] = []

    def work(index: int) -> None:
        try:
            service = ComputeOperationsService()
            barrier.wait(timeout=10)
            task = service.claim(f"worker-{index}", ["solver-a"], 60)
            if task is not None:
                with lock:
                    claimed.append(task)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    # 两个账号都从未超过运行上限，且全局队列继续前进
    assert len(claimed) == 3
    assert {task["id"] for task in claimed} == {alice_tasks[0]["id"], bob_tasks[0]["id"], bob_tasks[1]["id"]}
    for user, limit in (("alice", 1), ("bob", 2)):
        running = client.get(f"/api/compute/tasks?status=running&requested_by={user}").json()["items"]
        assert 0 < len(running) <= limit
    summary = client.get("/api/compute/summary").json()
    assert summary["states"]["running"] == 3
    assert summary["states"]["queued"] == 3
