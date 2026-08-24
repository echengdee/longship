from __future__ import annotations

import asyncio
import unittest

from longship.policies import (
    GuardedPolicyProvider,
    PolicyActionFrame,
    PolicyCandidate,
    PolicyCandidateRejected,
    PolicyGuardProfile,
    PolicyRequest,
    guard_candidate,
)


def request(*, deadline: float = 10.1) -> PolicyRequest:
    return PolicyRequest(
        call_id="call-1",
        model_binding_id="binding-1",
        lease_id="lease-1",
        lease_epoch=3,
        observation_version=7,
        deadline_monotonic=deadline,
        max_action_horizon_ms=100,
        resource_scope=("arm_motion",),
        payload={"observation": "synthetic"},
    )


def candidate(**overrides: object) -> PolicyCandidate:
    values: dict[str, object] = {
        "call_id": "call-1",
        "model_binding_id": "binding-1",
        "lease_id": "lease-1",
        "lease_epoch": 3,
        "observation_version": 7,
        "generated_at_monotonic": 10.0,
        "expires_at_monotonic": 10.05,
        "action_space_id": "test.action",
        "resource_scope": ("arm_motion",),
        "frames": (PolicyActionFrame(0, (0.1, -0.1)),),
    }
    values.update(overrides)
    return PolicyCandidate(**values)  # type: ignore[arg-type]


def profile() -> PolicyGuardProfile:
    return PolicyGuardProfile(
        action_space_id="test.action",
        action_dimension=2,
        permitted_resource_scope=("arm_motion",),
        max_action_horizon_ms=100,
        minimum_action_value=-1.0,
        maximum_action_value=1.0,
    )


class ImmediateBackend:
    def __init__(self, result: PolicyCandidate) -> None:
        self.result = result

    async def infer(self, policy_request: PolicyRequest) -> PolicyCandidate:
        return self.result


class SlowBackend:
    async def infer(self, policy_request: PolicyRequest) -> PolicyCandidate:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class PolicyGuardTests(unittest.TestCase):
    def test_accepts_correlated_bounded_candidate(self) -> None:
        result = guard_candidate(
            request(), candidate(), profile(), now_monotonic=10.01
        )
        self.assertEqual(result.frames[0].values, (0.1, -0.1))

    def test_rejects_stale_identity_and_scope_escalation(self) -> None:
        with self.assertRaisesRegex(PolicyCandidateRejected, "observation_version"):
            guard_candidate(
                request(),
                candidate(observation_version=8),
                profile(),
                now_monotonic=10.01,
            )
        with self.assertRaisesRegex(PolicyCandidateRejected, "resource scope"):
            guard_candidate(
                request(),
                candidate(resource_scope=("base_motion",)),
                profile(),
                now_monotonic=10.01,
            )

    def test_rejects_expiry_dimension_horizon_and_bounds(self) -> None:
        cases = (
            candidate(expires_at_monotonic=10.005),
            candidate(frames=(PolicyActionFrame(0, (0.1,)),)),
            candidate(frames=(PolicyActionFrame(100, (0.1, 0.2)),)),
            candidate(
                expires_at_monotonic=10.01,
                frames=(
                    PolicyActionFrame(0, (0.1, 0.2)),
                    PolicyActionFrame(99, (0.1, 0.2)),
                ),
            ),
            candidate(frames=(PolicyActionFrame(0, (1.1, 0.0)),)),
        )
        for result in cases:
            with self.subTest(result=result):
                with self.assertRaises(PolicyCandidateRejected):
                    guard_candidate(
                        request(), result, profile(), now_monotonic=10.01
                    )

    def test_request_payload_is_deeply_immutable(self) -> None:
        original = {"value": [{"nested": 1}]}
        policy_request = PolicyRequest(
            call_id="call",
            model_binding_id="binding",
            lease_id="lease",
            lease_epoch=1,
            observation_version=0,
            deadline_monotonic=1.0,
            max_action_horizon_ms=10,
            resource_scope=("base_motion",),
            payload=original,
        )
        original["value"][0]["nested"] = 2
        frozen = policy_request.payload["value"]
        self.assertIsInstance(frozen, tuple)
        self.assertEqual(frozen[0]["nested"], 1)  # type: ignore[index]
        with self.assertRaises(TypeError):
            policy_request.payload["new"] = 3  # type: ignore[index]


class GuardedPolicyProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_result_is_guarded(self) -> None:
        provider = GuardedPolicyProvider(
            ImmediateBackend(candidate()),
            profile(),
            lease_is_current=lambda value: True,
            clock=lambda: 10.01,
        )
        result = await provider.infer(request())
        self.assertEqual(result.call_id, "call-1")

    async def test_backend_timeout_fails_closed(self) -> None:
        times = iter((10.0,))
        provider = GuardedPolicyProvider(
            SlowBackend(),
            profile(),
            lease_is_current=lambda value: True,
            clock=lambda: next(times),
        )
        with self.assertRaisesRegex(PolicyCandidateRejected, "deadline"):
            await provider.infer(request(deadline=10.001))

    async def test_revoked_lease_fences_inflight_result(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        current = {"value": True}

        class ControlledBackend:
            async def infer(self, policy_request: PolicyRequest) -> PolicyCandidate:
                started.set()
                await release.wait()
                return candidate()

        provider = GuardedPolicyProvider(
            ControlledBackend(),
            profile(),
            lease_is_current=lambda value: current["value"],
            clock=lambda: 10.01,
        )
        task = asyncio.create_task(provider.infer(request()))
        await started.wait()
        current["value"] = False
        release.set()
        with self.assertRaisesRegex(PolicyCandidateRejected, "revoked"):
            await task


if __name__ == "__main__":
    unittest.main()
