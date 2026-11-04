import io

import pytest

pytest.importorskip("cryptography")

from mssb_coder import video_vault  # noqa: E402


@pytest.fixture(autouse=True)
def _key_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_VIDEO_KEY_FILE", str(tmp_path / "keys" / "video.key"))


SAMPLE = b"child video bytes " * 500_000  # 9MB: 여러 조각으로 나뉘는 크기


def test_encrypted_video_has_no_plaintext_and_round_trips(tmp_path):
    enc = video_vault.encrypt_upload(io.BytesIO(SAMPLE), tmp_path / "original.mov.enc")

    assert b"child video" not in enc.read_bytes()
    plain = video_vault.decrypt_file(enc, tmp_path / "plain.mov")
    assert plain.read_bytes() == SAMPLE


def test_empty_file_round_trips(tmp_path):
    enc = video_vault.encrypt_upload(io.BytesIO(b""), tmp_path / "empty.enc")
    assert video_vault.decrypt_file(enc, tmp_path / "out").read_bytes() == b""


def test_open_video_decrypts_temporarily_and_cleans_up(tmp_path):
    enc = video_vault.encrypt_upload(io.BytesIO(SAMPLE), tmp_path / "original.mov.enc")

    with video_vault.open_video(enc) as plain:
        assert plain.read_bytes() == SAMPLE
        assert plain != enc

    assert not plain.exists()
    assert enc.exists()


def test_open_video_passes_plain_files_through(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with video_vault.open_video(video) as plain:
        assert plain == video
    assert video.exists()


def test_truncated_or_tampered_file_fails(tmp_path):
    from cryptography.exceptions import InvalidTag

    enc = video_vault.encrypt_upload(io.BytesIO(SAMPLE), tmp_path / "original.mov.enc")
    data = enc.read_bytes()

    first_chunk_end = len(video_vault.MAGIC) + 7 + 4 + video_vault.CHUNK_SIZE + 16
    enc.write_bytes(data[:first_chunk_end])  # 뒷부분을 통째로 잘라냄
    with pytest.raises(InvalidTag):
        video_vault.decrypt_file(enc, tmp_path / "out")

    tampered = bytearray(data)
    tampered[100] ^= 1
    enc.write_bytes(bytes(tampered))
    with pytest.raises(InvalidTag):
        video_vault.decrypt_file(enc, tmp_path / "out")


def test_missing_key_gives_clear_error(tmp_path, monkeypatch):
    enc = video_vault.encrypt_upload(io.BytesIO(b"x"), tmp_path / "a.enc")
    monkeypatch.setenv("MSSB_VIDEO_KEY_FILE", str(tmp_path / "nowhere" / "video.key"))

    with pytest.raises(video_vault.VideoKeyMissingError):
        video_vault.decrypt_file(enc, tmp_path / "out")
