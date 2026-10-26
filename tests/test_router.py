import pytest

from jone.llm.base import LLMError, LLMRefusal
from jone.llm.router import REFUSAL_REPLY, LLMRouter, NoBackendAvailable, RouteInfo


class FakeBackend:
    def __init__(self, name, chunks=(), fail_after=None, error=LLMError("panne")):
        self.name = name
        self.chunks = list(chunks)
        self.fail_after = fail_after
        self.error = error
        self.calls = 0

    def stream(self, system, messages):
        self.calls += 1
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise self.error
            yield c
        if self.fail_after is not None and self.fail_after >= len(self.chunks):
            raise self.error

    def warmup(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def collect(router):
    info = RouteInfo()
    text = "".join(router.stream("sys", [{"role": "user", "content": "x"}], info))
    return text, info


def test_cloud_used_when_healthy():
    router = LLMRouter(FakeBackend("claude", ["Bon", "jour"]), FakeBackend("local", ["x"]))
    text, info = collect(router)
    assert text == "Bonjour" and info.backend == "claude" and not info.fallback


def test_falls_back_when_cloud_fails_before_first_token():
    local = FakeBackend("local", ["Salut"])
    router = LLMRouter(FakeBackend("claude", ["a"], fail_after=0), local)
    text, info = collect(router)
    assert text == "Salut" and info.backend == "local" and info.fallback
    assert info.errors == ["claude: panne"]


def test_cloud_skipped_during_cooldown_then_retried():
    clock = Clock()
    cloud = FakeBackend("claude", ["a"], fail_after=0)
    router = LLMRouter(cloud, FakeBackend("local", ["b"]), retry_cooldown_s=30, clock=clock)
    collect(router)
    clock.t = 10
    collect(router)
    assert cloud.calls == 1
    clock.t = 31
    collect(router)
    assert cloud.calls == 2


def test_partial_answer_kept_without_switching_model():
    local = FakeBackend("local", ["autre"])
    router = LLMRouter(FakeBackend("claude", ["Début", " suite"], fail_after=1), local)
    text, info = collect(router)
    assert text == "Début" and info.backend == "claude" and local.calls == 0


def test_refusal_gives_polite_reply():
    cloud = FakeBackend("claude", [], fail_after=0, error=LLMRefusal("non"))
    local = FakeBackend("local", ["x"])
    text, info = collect(LLMRouter(cloud, local))
    assert text == REFUSAL_REPLY and local.calls == 0


def test_no_backend_raises():
    with pytest.raises(NoBackendAvailable):
        collect(LLMRouter(FakeBackend("claude", [], fail_after=0), FakeBackend("local", [], fail_after=0)))
    with pytest.raises(NoBackendAvailable):
        collect(LLMRouter(None, None))


def test_local_only_mode():
    text, info = collect(LLMRouter(None, FakeBackend("local", ["ok"])))
    assert text == "ok" and info.backend == "local" and not info.fallback
