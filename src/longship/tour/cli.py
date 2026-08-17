from __future__ import annotations

import argparse
import asyncio
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

from longship.brain.codex_local import CodexLocalBrain
from longship.navigation.mock import MockNavigation

from .models import TourPlan
from .interaction import CommandKind
from .ports import ConsoleSpeaker
from .runtime import VoiceTourRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental Longship voice-tour V0")
    parser.add_argument("plan", type=Path, help="path to a longship.voice-tour.v0 JSON plan")
    parser.add_argument("--brain", choices=("off", "codex"), default="off")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--mock-travel-seconds", type=float, default=0.5)
    parser.add_argument("--speaker-label", default="Longship")
    parser.add_argument("--auto-start", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    plan = TourPlan.load(args.plan)
    speaker = ConsoleSpeaker(label=args.speaker_label)
    navigation = MockNavigation(travel_seconds=args.mock_travel_seconds)
    async with AsyncExitStack() as stack:
        brain = None
        if args.brain == "codex":
            temporary = tempfile.TemporaryDirectory(prefix="longship-codex-")
            stack.callback(temporary.cleanup)
            brain = await stack.enter_async_context(
                CodexLocalBrain(temporary.name, model=args.codex_model)
            )
        runtime = VoiceTourRuntime(plan, navigation, speaker, brain=brain)
        print("Commands: start/开始导览, pause/暂停, resume/恢复, next/下一站,")
        print("          repeat/重复, status/状态, cancel/取消, stop/停止")
        if args.auto_start:
            print(await runtime.start())
        command_tasks: set[asyncio.Task[str]] = set()
        stop_task: asyncio.Task[str] | None = None

        def completed(task: asyncio.Task[str]) -> None:
            nonlocal stop_task
            command_tasks.discard(task)
            if task is stop_task:
                stop_task = None
            if task.cancelled():
                return
            try:
                print(task.result())
            except Exception as exc:
                print(f"Command failed closed: {type(exc).__name__}")

        try:
            while True:
                try:
                    text = await asyncio.to_thread(input, "> ")
                except (EOFError, KeyboardInterrupt):
                    await runtime.cancel()
                    return 0
                routed = runtime.router.route(text)
                if (
                    routed.kind is CommandKind.STOP
                    and stop_task is not None
                    and not stop_task.done()
                ):
                    print("Protective stop is already in progress.")
                    continue
                if len(command_tasks) >= 8 and routed.kind is not CommandKind.STOP:
                    print("Too many pending requests; deterministic STOP remains available.")
                    continue
                # Separate tasks let reserved STOP overtake a slow Brain turn.
                task = asyncio.create_task(runtime.handle_text(text))
                if routed.kind is CommandKind.STOP:
                    stop_task = task
                command_tasks.add(task)
                task.add_done_callback(completed)
        finally:
            for task in command_tasks:
                task.cancel()
            if command_tasks:
                await asyncio.wait(command_tasks, timeout=1.0)


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
