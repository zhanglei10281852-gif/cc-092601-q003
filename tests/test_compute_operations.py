from __future__ import annotations

import threading
from datetime import UTC, datetime

from app.compute.service import ComputeOperationsService
from app.core.clock import FrozenClock
from app.database import close_connection, get_connection, transaction


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


def _start_compute_service(tmp_path, monkeypatch):
    """绕过 HTTP，直接在临时 SQLite 文件上构建计算服务（供多线程共享）。"""
    from app.database import close_connection, init_db

    monkeypatch.setenv("TOWNSHIP_DATABASE_PATH", str(tmp_path / "quota-concurrency.db"))
    close_connection()
    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 6, 0, tzinfo=UTC))
    service = ComputeOperationsService(get_connection(), clock)
    service.create_template(TEMPLATE, "administrator")
    return service, clock


def test_claim_skips_full_user_and_keeps_global_queue_moving(tmp_path, monkeypatch):
    """队首账号已满时，领取应跳过它继续为后面的用户派单；完成后名额立即释放。"""
    service, _ = _start_compute_service(tmp_path, monkeypatch)
    service.set_quota(
        {"subject_type": "user", "subject_key": "user-a", "max_queued": 10, "max_running": 1, "daily_submissions": 100},
        "administrator",
    )
    a1 = service.submit(submit_payload("seq-a1", user="user-a"))
    a2 = service.submit(submit_payload("seq-a2", user="user-a"))
    b1 = service.submit(submit_payload("seq-b1", user="user-b"))
    first = service.claim("w1", ["solver-a"], 60)
    second = service.claim("w2", ["solver-a"], 60)
    third = service.claim("w3", ["solver-a"], 60)
    assert [item["id"] for item in (first, second) if item] == [a1["id"], b1["id"]]
    assert third is None  # user-a 已满，a2 不得被领取
    assert service.get_task(a2["id"])["status"] == "queued"
    service.complete(a1["id"], "w1", {"value": 1}, {})
    after_release = service.claim("w4", ["solver-a"], 60)
    assert after_release and after_release["id"] == a2["id"]
    service.complete(b1["id"], "w2", {"value": 2}, {})
    service.complete(a2["id"], "w4", {"value": 3}, {})
    close_connection()


def test_concurrent_claims_never_exceed_running_quota(tmp_path, monkeypatch):
    """两名用户、多名工作者、固定提交顺序下的并发领取与名额释放全链路。"""
    service, clock = _start_compute_service(tmp_path, monkeypatch)
    for user in ("user-a", "user-b"):
        service.set_quota(
            {"subject_type": "user", "subject_key": user, "max_queued": 10, "max_running": 1, "daily_submissions": 100},
            "administrator",
        )

    # 固定顺序：user-a 连续四条排在队首，其后是 user-b 的任务。
    order = [
        ("wave-a1", "user-a"), ("wave-a2", "user-a"), ("wave-a3", "user-a"), ("wave-a4", "user-a"),
        ("wave-b1", "user-b"), ("wave-b2", "user-b"),
    ]
    tasks = {key: service.submit(submit_payload(key, user=user)) for key, user in order}

    snapshots: list[tuple[int, int]] = []
    stop_auditor = threading.Event()

    def audit() -> None:
        connection = get_connection()
        while not stop_auditor.wait(0.002):
            rows = connection.execute(
                "SELECT requested_by,COUNT(*) AS amount FROM compute_tasks WHERE status='running' GROUP BY requested_by"
            ).fetchall()
            counts = {row["requested_by"]: int(row["amount"]) for row in rows}
            snapshots.append((counts.get("user-a", 0), counts.get("user-b", 0)))

    auditor = threading.Thread(target=audit)
    auditor.start()
    before_quota_change = 0
    try:
        # 第一波：6 名工作者被同一屏障同时释放并发领取。
        claims: dict[str, dict | None] = {}
        claims_lock = threading.Lock()
        barrier = threading.Barrier(len(order))

        def worker(worker_id: str) -> None:
            worker_service = ComputeOperationsService(get_connection(), clock)
            barrier.wait()
            task = worker_service.claim(worker_id, ["solver-a"], 60)
            with claims_lock:
                claims[worker_id] = task

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(len(order))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        claimed_ids = {task["id"] for task in claims.values() if task}
        # 只有队首 a1 与跳过满额账号后选中的 b1 被领取，其余工作者空手而回。
        assert claimed_ids == {tasks["wave-a1"]["id"], tasks["wave-b1"]["id"]}

        def owner_of(task_id: int) -> str:
            return next(wid for wid, task in claims.items() if task and task["id"] == task_id)

        # 完成立即释放 a 的名额 -> a2 领取；取消 running 任务也立即释放 -> a3 领取。
        service.complete(tasks["wave-a1"]["id"], owner_of(tasks["wave-a1"]["id"]), {"value": 1}, {})
        a2 = service.claim("main-a2", ["solver-a"], 60)
        assert a2 and a2["id"] == tasks["wave-a2"]["id"]
        cancelled = service.cancel(tasks["wave-a2"]["id"], "administrator", "项目暂停")
        assert cancelled["status"] == "cancel_requested"
        a3 = service.claim("main-a3", ["solver-a"], 60)
        assert a3 and a3["id"] == tasks["wave-a3"]["id"]

        # 失败（可重试，带退避）立即释放 b 的名额：b1 退避未到，队列前进到 b2。
        failed = service.fail(tasks["wave-b1"]["id"], owner_of(tasks["wave-b1"]["id"]), "numeric_error", "不收敛", True)
        assert failed["status"] == "queued"
        b2 = service.claim("main-b2", ["solver-a"], 60)
        assert b2 and b2["id"] == tasks["wave-b2"]["id"]

        # 配额上调只影响后续领取，不中断已有运行：a3 仍在运行，a4 现在可以并行领取。
        before_quota_change = len(snapshots)
        service.set_quota(
            {"subject_type": "user", "subject_key": "user-a", "max_queued": 10, "max_running": 2, "daily_submissions": 100},
            "administrator",
        )
        a4 = service.claim("main-a4", ["solver-a"], 60)
        assert a4 and a4["id"] == tasks["wave-a4"]["id"]
        assert service.get_task(tasks["wave-a3"]["id"])["status"] == "running"
        assert service.get_task(tasks["wave-a4"]["id"])["lease_owner"] == "main-a4"
        service.complete(tasks["wave-a3"]["id"], "main-a3", {"value": 3}, {})
        service.complete(tasks["wave-a4"]["id"], "main-a4", {"value": 4}, {})

        # 配额下调同样不强制中断已有运行，且后续领取按新上限执行。
        service.set_quota(
            {"subject_type": "user", "subject_key": "user-a", "max_queued": 10, "max_running": 1, "daily_submissions": 100},
            "administrator",
        )

        # 租约恢复立即释放名额：推进时钟超过 b2 租约后恢复，随后任务可被再次领取。
        clock.advance(seconds=61)
        recovery = service.recover_expired()
        assert recovery["recovered"] == [tasks["wave-b2"]["id"]]
        # b1 退避已结束且 id 更小排在 b2 之前，先被重新领取（第二次尝试）。
        b1_again = service.claim("main-b1-retry", ["solver-a"], 60)
        assert b1_again and b1_again["id"] == tasks["wave-b1"]["id"] and b1_again["attempt_count"] == 2
        service.complete(tasks["wave-b1"]["id"], "main-b1-retry", {"value": 5}, {})
        b2_again = service.claim("main-b2-retry", ["solver-a"], 60)
        assert b2_again and b2_again["id"] == tasks["wave-b2"]["id"]
        service.complete(tasks["wave-b2"]["id"], "main-b2-retry", {"value": 6}, {})
    finally:
        stop_auditor.set()
        auditor.join()

    # 审计线程观察到的所有已提交状态中，每个账号从未超过当时的运行上限。
    assert snapshots, "审计线程应采集到运行中快照"
    assert max((a for a, _ in snapshots[:before_quota_change]), default=0) <= 1
    assert max((b for _, b in snapshots[:before_quota_change]), default=0) <= 1
    assert max((a for a, _ in snapshots[before_quota_change:]), default=0) <= 2
    assert max((b for _, b in snapshots[before_quota_change:]), default=0) <= 1

    final_states = service.summary()["states"]
    assert final_states.get("running", 0) == 0
    assert final_states.get("queued", 0) == 0
    assert final_states.get("succeeded") == 5
    assert final_states.get("cancel_requested") == 1
    close_connection()
