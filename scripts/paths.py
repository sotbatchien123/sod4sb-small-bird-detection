"""Paths written into result files (JSON, checkpoints, dumped configs) are always relative to the repository root,
never absolute, so outputs are portable and reveal nothing about the machine they were made on.
Absolute paths are still used at run time to open files; only what is *saved* goes through these helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rel(p):
    """Repository-relative path with '/' separators; outside the repository only the file name is kept."""
    p = Path(p)
    try:
        return (p if not p.is_absolute() else p.resolve().relative_to(ROOT)).as_posix()
    except ValueError:
        return p.name


def dump_config(cfg, path):
    """mmcv Config.dump, with the absolute repository prefix that '{{ fileDirname }}' expands to removed."""
    cfg.dump(str(path))
    text = Path(path).read_text(encoding='utf-8')
    for prefix in (f'{ROOT.as_posix()}/configs/../', f'{ROOT.as_posix()}/', f'{ROOT}\\configs\\..\\', f'{ROOT}\\'):
        text = text.replace(prefix, '')
    Path(path).write_bytes(text.encode('utf-8'))   # Python 3.9: write_text has no newline argument
