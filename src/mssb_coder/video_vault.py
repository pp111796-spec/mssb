"""원본 영상 암호화 — 앱으로 올린 영상을 AES-256-GCM으로 암호화해 저장하고, 분석할 때만 임시로 푼다.

키는 데이터 폴더(MSSB_DATA_DIR)와 OneDrive 밖의 사용자 폴더(~/.mssb/video.key)에 둔다. 데이터 폴더만
복사돼 나가도 영상을 볼 수 없게 하려는 것이다. **키 파일을 잃으면 암호화된 영상은 복구할 수 없다.**

파일 형식: MAGIC + 7바이트 난수 + [4바이트 길이 + 암호문]... — 조각마다 (난수 + 순번 + 마지막 표시)를
nonce로 써서 조각 순서 바꾸기나 파일 뒷부분 잘라내기도 복호화 실패로 잡힌다.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"MSSBENC1"
CHUNK_SIZE = 4 * 1024 * 1024
ENCRYPTED_SUFFIX = ".enc"


class VideoKeyMissingError(RuntimeError):
    pass


def key_path() -> Path:
    override = os.environ.get("MSSB_VIDEO_KEY_FILE")
    return Path(override) if override else Path.home() / ".mssb" / "video.key"


def _load_key(create: bool) -> bytes:
    path = key_path()
    if path.exists():
        return path.read_bytes()
    if not create:
        raise VideoKeyMissingError(f"영상 암호화 키 파일이 없어 영상을 열 수 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    path.write_bytes(key)
    return key


def _nonce(prefix: bytes, counter: int, last: bool) -> bytes:
    return prefix + counter.to_bytes(4, "big") + (b"\x01" if last else b"\x00")


def _encrypt_stream(source: BinaryIO, out_path: Path) -> Path:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

    aead = AESGCM(_load_key(create=True))
    prefix = os.urandom(7)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as out:
        out.write(MAGIC + prefix)
        counter = 0
        chunk = source.read(CHUNK_SIZE)
        while True:
            following = source.read(CHUNK_SIZE)
            sealed = aead.encrypt(_nonce(prefix, counter, last=not following), chunk, None)
            out.write(len(sealed).to_bytes(4, "big") + sealed)
            if not following:
                break
            chunk, counter = following, counter + 1
    tmp_path.replace(out_path)
    return out_path


def encrypt_file(src_path: Path, out_path: Path) -> Path:
    with src_path.open("rb") as source:
        return _encrypt_stream(source, out_path)


def encrypt_upload(source: BinaryIO, out_path: Path) -> Path:
    """업로드 스트림을 평문 파일을 남기지 않고 바로 암호화해 저장한다."""
    return _encrypt_stream(source, out_path)


def decrypt_file(src_path: Path, out_path: Path) -> Path:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

    aead = AESGCM(_load_key(create=False))
    with src_path.open("rb") as source, out_path.open("wb") as out:
        header = source.read(len(MAGIC) + 7)
        if not header.startswith(MAGIC):
            raise ValueError(f"암호화된 영상 파일 형식이 아닙니다: {src_path}")
        prefix = header[len(MAGIC):]
        counter = 0
        length = source.read(4)
        while length:
            sealed = source.read(int.from_bytes(length, "big"))
            following = source.read(4)
            out.write(aead.decrypt(_nonce(prefix, counter, last=not following), sealed, None))
            length, counter = following, counter + 1
    return out_path


def is_encrypted(path: Path) -> bool:
    return path.name.endswith(ENCRYPTED_SUFFIX)


@contextmanager
def open_video(video_path: Path) -> Iterator[Path]:
    """ffmpeg에 넘길 평문 경로를 준다. 암호화 파일이면 같은 폴더에 임시로 풀고, 끝나면 지운다."""
    video_path = Path(video_path)
    if not is_encrypted(video_path):
        yield video_path
        return
    original_name = video_path.name[: -len(ENCRYPTED_SUFFIX)]
    plain_path = video_path.with_name(f"decrypted_tmp_{original_name}")
    try:
        decrypt_file(video_path, plain_path)
        yield plain_path
    finally:
        plain_path.unlink(missing_ok=True)
