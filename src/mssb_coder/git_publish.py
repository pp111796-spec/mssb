"""results/ 게시용 git 처리 — 결과 파일 하나만 커밋·푸시한다.

아직 푸시하지 않은 다른 커밋이 있으면 게시를 거부한다. 그대로 `git push`를 하면 그 커밋(코드 변경 등)이
결과 파일과 함께 공유 저장소에 올라가기 때문이다. runner를 주입받아 실제 git 없이 테스트할 수 있다.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GIT_TIMEOUT_S = 60

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class PublishOutcome:
    ok: bool
    message: str


def _git(runner: Runner, repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return runner(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT_S,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _detail(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or "").strip()


def publish_file(repo_root: Path, file_path: Path, message: str, runner: Runner = subprocess.run) -> PublishOutcome:
    rel = file_path.resolve().relative_to(repo_root.resolve()).as_posix()
    committed = False
    try:
        ahead = _git(runner, repo_root, "rev-list", "--count", "@{u}..HEAD")
        if ahead.returncode != 0:
            return PublishOutcome(False, f"원격 저장소 상태를 확인하지 못했습니다.\n{_detail(ahead)}")
        ahead_count = int(ahead.stdout.strip() or "0")
        if ahead_count > 0:
            return PublishOutcome(
                False,
                f"아직 공유 저장소에 올리지 않은 다른 커밋이 {ahead_count}개 있어 게시를 멈췄습니다. "
                "그 커밋까지 함께 올라가는 것을 막기 위해서입니다. 코드 변경을 먼저 정리(푸시)한 뒤 다시 게시하세요.",
            )

        add = _git(runner, repo_root, "add", "--", rel)
        if add.returncode != 0:
            return PublishOutcome(False, f"결과 파일을 git에 추가하지 못했습니다.\n{_detail(add)}")

        if _git(runner, repo_root, "diff", "--cached", "--quiet", "--", rel).returncode == 0:
            return PublishOutcome(True, "이미 게시된 내용과 같아서 새로 올릴 것이 없습니다.")

        commit = _git(runner, repo_root, "commit", "--only", "-m", message, "--", rel)
        if commit.returncode != 0:
            return PublishOutcome(False, f"커밋하지 못했습니다.\n{_detail(commit)}")
        committed = True

        push = _git(runner, repo_root, "push")
        if push.returncode != 0:
            _git(runner, repo_root, "reset", "--soft", "HEAD~1")
            return PublishOutcome(
                False,
                "공유 저장소에 올리지 못해 방금 만든 커밋을 되돌렸습니다. 인터넷 연결이나 GitHub 로그인 상태를 "
                f"확인한 뒤 다시 게시하세요.\n{_detail(push)}",
            )
        return PublishOutcome(True, "게시 완료 — 1~2분 뒤 공유 사이트에 반영됩니다.")
    except FileNotFoundError:
        return PublishOutcome(False, "git이 설치되어 있지 않거나 찾을 수 없습니다. git을 설치한 뒤 다시 시도하세요.")
    except subprocess.TimeoutExpired:
        if committed:
            try:
                _git(runner, repo_root, "reset", "--soft", "HEAD~1")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return PublishOutcome(
            False,
            f"git 응답이 {GIT_TIMEOUT_S}초 안에 오지 않아 게시를 멈췄습니다. 인터넷 연결이나 GitHub 로그인 상태를 확인하세요.",
        )
