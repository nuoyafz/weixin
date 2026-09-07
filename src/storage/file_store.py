"""文件存储：把导入的文档复制到 data 目录，生成引用索引，支持列出/删除。

引用索引持久化在 file_refs 表。遵循"每次操作新建连接"模式。
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

from . import db

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DOCS_DIR = _DATA_DIR / "files"
_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _norm(target: str) -> Path:
    p = Path(target).expanduser().resolve()
    return p


def store_file(
    source: str,
    kind: str = "doc",
    name: Optional[str] = None,
    dedupe: bool = True,
) -> dict:
    """把源文档复制到 data/files 目录并写入引用索引；返回索引记录。

    dedupe=True 时按文件大小+内容哈希跳过重复文件（复用已有记录）。
    """
    src = _norm(source)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")

    display_name = (name or src.name).strip() or src.name
    size = src.stat().st_size

    if dedupe:
        digest = _hash(src)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM file_refs WHERE name=? AND size=? LIMIT 1",
                (display_name, size),
            ).fetchone()
        # 内容哈希校验：若同名同大小且哈希相同，复用已有记录
        if row and _hash(Path(row["stored_path"])) == digest:
            return dict(row)

    stored_name = _safe_name(display_name, src)
    dest = _DOCS_DIR / stored_name
    shutil.copy2(src, dest)
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO file_refs(source_path,stored_path,name,size,kind,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (str(src), str(dest), display_name, size, kind, db.now_str()),
        )
        rid = int(cur.lastrowid)
    return {
        "id": rid,
        "source_path": str(src),
        "stored_path": str(dest),
        "name": display_name,
        "size": size,
        "kind": kind,
        "created_at": db.now_str(),
    }


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(display_name: str, src: Path) -> str:
    """生成落盘文件名：含扩展名、避免重名（追加序号）。"""
    stem = src.stem or display_name
    suffix = src.suffix or Path(display_name).suffix or ""
    candidate = f"{stem}{suffix}"
    dest = _DOCS_DIR / candidate
    if not dest.exists():
        return candidate
    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if not (_DOCS_DIR / candidate).exists():
            return candidate
        i += 1


def list_files(kind: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[dict]:
    with db.get_conn() as conn:
        if kind:
            rows = conn.execute(
                "SELECT * FROM file_refs WHERE kind=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM file_refs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]


def get_file(ref_id: int) -> Optional[dict]:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM file_refs WHERE id=?", (ref_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_file(ref_id: int) -> bool:
    """删除引用索引及对应存储副本；返回是否存在该记录。"""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT stored_path FROM file_refs WHERE id=?", (ref_id,)
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM file_refs WHERE id=?", (ref_id,))
    path = Path(row["stored_path"])
    if path.exists():
        path.unlink()
    return True


def clean_orphans() -> int:
    """删除存储目录中无索引引用的孤儿文件，返回删除数量。"""
    removed = 0
    with db.get_conn() as conn:
        tracked = {
            Path(r["stored_path"])
            for r in conn.execute("SELECT stored_path FROM file_refs").fetchall()
        }
    for p in _DOCS_DIR.iterdir():
        if p.is_file() and p.resolve() not in tracked:
            p.unlink()
            removed += 1
    return removed


class FileStore:
    """文件存储：包装模块级函数为类接口。"""

    def __init__(self):
        self._logs: list = []

    def append_log(self, message: str, detail: str = "") -> None:
        """追加运行时日志（同时落盘到本地 logs/，受 LocalLogger 管理）。"""
        self._logs.append({"message": message, "detail": detail, "time": __import__("time").time()})
        try:
            from ..common.local_logger import LocalLogger
            LocalLogger.log("store", (message + (" " + detail if detail else "")).strip())
        except Exception:
            pass

    def save_report(self, report: dict, send_result: dict) -> None:
        """保存发送报告。"""
        pass

    def store_file(self, source: str, kind: str = "doc",
                   name: Optional[str] = None, dedupe: bool = True) -> dict:
        return store_file(source, kind=kind, name=name, dedupe=dedupe)

    def list_files(self, kind: Optional[str] = None,
                   limit: int = 200, offset: int = 0) -> List[dict]:
        return list_files(kind=kind, limit=limit, offset=offset)

    def get_file(self, ref_id: int) -> Optional[dict]:
        return get_file(ref_id)

    def delete_file(self, ref_id: int) -> bool:
        return delete_file(ref_id)

    def clean_orphans(self) -> int:
        return clean_orphans()