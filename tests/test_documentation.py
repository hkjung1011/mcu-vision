import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _markdown_files() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "THIRD_PARTY_NOTICES.md",
        PROJECT_ROOT / "reports" / "README.md",
        PROJECT_ROOT / "weights" / "README.md",
    ]
    candidates.extend((PROJECT_ROOT / "docs").rglob("*.md"))
    candidates.extend((PROJECT_ROOT / "reports" / "methodology").glob("*.md"))
    return sorted({path for path in candidates if path.exists()})


def test_local_markdown_links_exist() -> None:
    missing: list[str] = []
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for document in _markdown_files():
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            linked = (document.parent / target).resolve()
            if not linked.exists():
                missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
    assert not missing, "Missing local Markdown links:\n" + "\n".join(missing)


def test_repository_documentation_has_no_local_user_path() -> None:
    offenders = []
    for document in _markdown_files():
        text = document.read_text(encoding="utf-8").lower()
        if "c:\\users\\" in text or "c:/users/" in text:
            offenders.append(str(document.relative_to(PROJECT_ROOT)))
    assert not offenders, f"Local Windows user paths found in documentation: {offenders}"
