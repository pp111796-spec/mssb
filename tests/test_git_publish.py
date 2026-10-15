import subprocess
from pathlib import Path

from mssb_coder.git_publish import publish_file


class FakeGit:
    """git 하위 명령(첫 인자)별로 (returncode, stdout)을 돌려주는 가짜 runner."""

    def __init__(self, responses: dict[str, tuple[int, str]] | None = None, raise_on: dict[str, Exception] | None = None):
        self.responses = responses or {}
        self.raise_on = raise_on or {}
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        assert kwargs["timeout"] > 0
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        self.calls.append(args)
        sub = args[1]
        if sub in self.raise_on:
            raise self.raise_on[sub]
        code, out = self.responses.get(sub, (0, ""))
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="")

    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls]


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    results.mkdir()
    file_path = results / "s1.json"
    file_path.write_text("{}", encoding="utf-8")
    return tmp_path, file_path


def test_publish_commits_only_the_results_file_and_pushes(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit({"rev-list": (0, "0\n"), "diff": (1, "")})

    outcome = publish_file(repo, file_path, "세션 결과 게시: s1", runner=fake)

    assert outcome.ok
    commit = next(c for c in fake.calls if c[1] == "commit")
    assert "--only" in commit and commit[-2:] == ["--", "results/s1.json"]
    assert fake.subcommands() == ["rev-list", "add", "diff", "commit", "push"]


def test_publish_refuses_when_other_commits_are_unpushed(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit({"rev-list": (0, "1\n")})

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert not outcome.ok
    assert "1개" in outcome.message
    assert fake.subcommands() == ["rev-list"]


def test_publish_reports_nothing_to_commit_without_committing(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit({"rev-list": (0, "0"), "diff": (0, "")})

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert outcome.ok
    assert "commit" not in fake.subcommands()
    assert "push" not in fake.subcommands()


def test_publish_undoes_local_commit_when_push_fails(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit({"rev-list": (0, "0"), "diff": (1, ""), "push": (1, "rejected")})

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert not outcome.ok
    assert fake.calls[-1] == ["git", "reset", "--soft", "HEAD~1"]


def test_publish_undoes_commit_when_push_times_out(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit(
        {"rev-list": (0, "0"), "diff": (1, "")},
        raise_on={"push": subprocess.TimeoutExpired(["git", "push"], 60)},
    )

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert not outcome.ok
    assert "60초" in outcome.message
    assert fake.calls[-1] == ["git", "reset", "--soft", "HEAD~1"]


def test_publish_explains_missing_git(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit(raise_on={"rev-list": FileNotFoundError("git")})

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert not outcome.ok
    assert "git" in outcome.message


def test_publish_reports_missing_upstream(tmp_path):
    repo, file_path = _paths(tmp_path)
    fake = FakeGit({"rev-list": (128, "no upstream")})

    outcome = publish_file(repo, file_path, "msg", runner=fake)

    assert not outcome.ok
    assert fake.subcommands() == ["rev-list"]
