"""Tests for knowlin_mcp server and utils modules.

Tests verify:
- TCP socket communication
- Port file management
- Server status detection
- Cross-platform compatibility
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _send_server_request(server, request):
    client, server_conn = socket.socketpair()
    thread = threading.Thread(target=server.handle_client, args=(server_conn,))
    thread.start()
    try:
        client.sendall(json.dumps(request).encode("utf-8"))
        client.shutdown(socket.SHUT_WR)

        chunks = []
        while True:
            chunk = client.recv(1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        client.close()
        thread.join(timeout=3.0)

    return json.loads(b"".join(chunks).decode("utf-8"))


# =============================================================================
# Port File Tests
# =============================================================================


class TestPortFileManagement:
    """Tests for port file operations."""

    def test_get_kb_port_file_returns_path(self, tmp_path):
        from knowlin_mcp.platform import get_kb_port_file

        project = tmp_path / "test-project"
        project.mkdir()

        result = get_kb_port_file(project)
        assert isinstance(result, Path)
        assert result.suffix == ".port"

    def test_port_file_contains_project_hash(self, tmp_path):
        from knowlin_mcp.platform import get_kb_port_file, get_project_hash

        project = tmp_path / "test-project"
        project.mkdir()

        result = get_kb_port_file(project)
        project_hash = get_project_hash(project)
        assert project_hash in result.stem

    def test_get_kb_token_file_returns_path(self, tmp_path):
        from knowlin_mcp.platform import get_kb_token_file

        project = tmp_path / "test-project"
        project.mkdir()

        result = get_kb_token_file(project)
        assert isinstance(result, Path)
        assert result.suffix == ".token"


class TestGetServerPort:
    """Tests for get_server_port()."""

    def test_returns_none_when_no_port_file(self, tmp_path):
        from knowlin_mcp.utils import get_server_port

        project = tmp_path / "project"
        project.mkdir()

        nonexistent = tmp_path / "nonexistent.port"
        with patch("knowlin_mcp.utils.get_kb_port_file", return_value=nonexistent):
            result = get_server_port(project)
            assert result is None

    def test_reads_port_from_file(self, tmp_path):
        from knowlin_mcp.utils import get_server_port

        project = tmp_path / "project"
        project.mkdir()

        port_file = tmp_path / "test.port"
        port_file.write_text("14567")

        with patch("knowlin_mcp.utils.get_kb_port_file", return_value=port_file):
            result = get_server_port(project)
            assert result == 14567


# =============================================================================
# Server Status Tests
# =============================================================================


class TestIsServerRunning:
    """Tests for is_server_running()."""

    def test_returns_false_when_no_port_file(self, tmp_path):
        from knowlin_mcp.utils import is_server_running

        project = tmp_path / "project"
        project.mkdir()

        with patch("knowlin_mcp.utils.get_server_port", return_value=None):
            result = is_server_running(project)
            assert result is False

    def test_returns_false_when_connection_fails(self, tmp_path):
        from knowlin_mcp.utils import is_server_running

        project = tmp_path / "project"
        project.mkdir()

        with patch("knowlin_mcp.utils.get_server_port", return_value=59999):
            result = is_server_running(project, timeout=0.1)
            assert result is False


class TestTcpCommunication:
    """Tests for TCP socket communication."""

    def test_send_command_returns_none_without_port(self, tmp_path):
        from knowlin_mcp.utils import send_command

        project = tmp_path / "project"
        project.mkdir()

        with patch("knowlin_mcp.utils.get_server_port", return_value=None):
            result = send_command(project, {"cmd": "ping"})
            assert result is None

    def test_tcp_server_communication(self):
        """Test TCP server/client communication works."""
        server_received = []

        def echo_server(port):
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", port))
            server.listen(1)
            server.settimeout(2.0)
            try:
                conn, _ = server.accept()
                data = conn.recv(1024).decode()
                server_received.append(data)
                response = json.dumps({"pong": True})
                conn.sendall(response.encode())
                conn.close()
            except socket.timeout:
                pass
            finally:
                server.close()

        # Find an available port
        test_port = 54321
        for offset in range(10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", test_port + offset))
                sock.close()
                test_port = test_port + offset
                break
            except OSError:
                continue

        server_thread = threading.Thread(target=echo_server, args=(test_port,))
        server_thread.start()
        time.sleep(0.1)

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(1.0)
            client.connect(("127.0.0.1", test_port))
            client.sendall(json.dumps({"cmd": "ping"}).encode())
            response = client.recv(1024).decode()
            client.close()

            assert '"pong"' in response
            assert len(server_received) == 1
        finally:
            server_thread.join(timeout=3.0)


class TestKnowledgeServerAdd:
    """Tests for KnowledgeServer._cmd_add()."""

    def test_cmd_add_returns_error_when_entry_is_rejected(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.db = MagicMock()
        server.db.add.return_value = ""

        response = server._cmd_add({"entry": {"title": "Bad entry"}})

        assert "error" in response
        assert "rejected" in response["error"].lower()


class TestKnowledgeServerSearch:
    """Tests for KnowledgeServer._cmd_search()."""

    def test_cmd_search_uses_multi_source_for_single_source_filter(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.db = MagicMock()
        server.ms = MagicMock()
        server.ms.search.return_value = [{"id": "doc-1", "_source": "docs"}]

        response = server._cmd_search({"query": "auth", "sources": ["docs"], "limit": 3})

        server.ms.search.assert_called_once_with(
            "auth",
            sources=["docs"],
            limit=3,
            date_from=None,
            date_to=None,
            entry_type=None,
            branch=None,
        )
        server.db.search.assert_not_called()
        assert response["multi_source"] is True
        assert response["results"] == [{"id": "doc-1", "_source": "docs"}]

    def test_cmd_search_clamps_limit(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.db = MagicMock()
        server.db.search.return_value = []

        server._cmd_search({"query": "auth", "limit": 99999})

        server.db.search.assert_called_once_with(
            "auth",
            50,
            date_from=None,
            date_to=None,
            entry_type=None,
            branch=None,
        )


class TestKnowledgeServerLimits:
    """Tests for TCP command limit clamping."""

    def test_cmd_recent_clamps_limit(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.db = MagicMock()
        server.db.get_recent_important.return_value = []

        server._cmd_recent({"limit": 99999})

        server.db.get_recent_important.assert_called_once_with(50)

    def test_cmd_search_by_date_clamps_limit(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.db = MagicMock()
        server.db.search_by_date.return_value = []

        server._cmd_search_by_date({"start": "2026-01-01", "limit": 99999})

        server.db.search_by_date.assert_called_once_with("2026-01-01", None, 50)


class TestKnowledgeServerAuthentication:
    """Tests for per-request TCP authentication."""

    def test_handle_client_rejects_missing_token(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.token = "secret-token"

        response = _send_server_request(server, {"cmd": "ping"})

        assert response == {"status": "error", "error": "authentication required"}

    def test_handle_client_sanitizes_internal_errors(self, tmp_path):
        from knowlin_mcp.server import KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.token = "secret-token"
        sensitive_path = tmp_path / "private" / "secret.txt"

        def explode(request):
            raise RuntimeError(f"failed while reading {sensitive_path}")

        server._handlers["explode"] = explode

        response = _send_server_request(server, {"cmd": "explode", "token": "secret-token"})

        assert response == {"status": "error", "error": "internal server error"}
        assert str(sensitive_path) not in json.dumps(response)

    def test_send_command_includes_token(self, tmp_path):
        from knowlin_mcp.utils import send_command

        project = tmp_path / "project"
        project.mkdir()
        token_file = tmp_path / "test.token"
        token_file.write_text("secret-token")
        received = []

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        def auth_server():
            conn, _ = server_sock.accept()
            chunks = []
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                chunks.append(chunk)
            request = json.loads(b"".join(chunks).decode("utf-8"))
            received.append(request)
            if request.get("token") == "secret-token":
                conn.sendall(json.dumps({"pong": True}).encode("utf-8"))
            else:
                conn.sendall(
                    json.dumps({"status": "error", "error": "authentication required"}).encode(
                        "utf-8"
                    )
                )
            conn.close()
            server_sock.close()

        thread = threading.Thread(target=auth_server)
        thread.start()

        try:
            with (
                patch("knowlin_mcp.utils.get_server_port", return_value=port),
                patch("knowlin_mcp.utils.get_kb_token_file", return_value=token_file),
            ):
                response = send_command(project, {"cmd": "ping"})
        finally:
            thread.join(timeout=3.0)

        assert response == {"pong": True}
        assert received == [{"cmd": "ping", "token": "secret-token"}]


# =============================================================================
# KB Initialization Tests
# =============================================================================


class TestIsKbInitialized:
    """Tests for is_kb_initialized()."""

    def test_returns_false_when_no_kb_dir(self, tmp_path):
        from knowlin_mcp.utils import is_kb_initialized

        project = tmp_path / "project"
        project.mkdir()
        assert is_kb_initialized(project) is False

    def test_returns_true_when_kb_dir_exists(self, tmp_path):
        from knowlin_mcp.utils import is_kb_initialized

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()
        assert is_kb_initialized(project) is True


# =============================================================================
# Stale File Cleanup Tests
# =============================================================================


class TestCleanStaleSocket:
    """Tests for clean_stale_socket()."""

    def test_returns_false_when_no_port_file(self, tmp_path):
        from knowlin_mcp.utils import clean_stale_socket

        project = tmp_path / "project"
        project.mkdir()

        with patch("knowlin_mcp.utils.get_kb_port_file") as mock_port_file:
            mock_port_file.return_value = tmp_path / "nonexistent.port"
            result = clean_stale_socket(project)
            assert result is False

    def test_cleans_stale_files_when_server_not_running(self, tmp_path):
        from knowlin_mcp.utils import clean_stale_socket

        project = tmp_path / "project"
        project.mkdir()

        port_file = tmp_path / "test.port"
        pid_file = tmp_path / "test.pid"
        token_file = tmp_path / "test.token"
        port_file.write_text("59999")
        pid_file.write_text("999999")
        token_file.write_text("secret-token")

        with (
            patch("knowlin_mcp.utils.get_kb_port_file", return_value=port_file),
            patch("knowlin_mcp.utils.get_kb_pid_file", return_value=pid_file),
            patch("knowlin_mcp.utils.get_kb_token_file", return_value=token_file),
            patch("knowlin_mcp.utils.is_server_running", return_value=False),
        ):
            result = clean_stale_socket(project)
            assert result is True
            assert not port_file.exists()
            assert not pid_file.exists()
            assert not token_file.exists()


# =============================================================================
# Cross-Platform Tests
# =============================================================================


class TestCrossPlatform:
    """Tests for cross-platform compatibility."""

    def test_uses_tcp_not_unix_socket(self):
        import inspect

        from knowlin_mcp.utils import is_server_running

        source = inspect.getsource(is_server_running)
        assert "AF_INET" in source
        assert "AF_UNIX" not in source

    def test_host_is_localhost(self):
        from knowlin_mcp.platform import HOST

        assert HOST == "127.0.0.1"


class TestRuntimeDirSecurity:
    """Tests for runtime directory security checks."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits differ on Windows")
    def test_runtime_dir_has_700_permissions(self, tmp_path, monkeypatch):
        from knowlin_mcp import platform

        monkeypatch.setattr(platform.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(platform.getpass, "getuser", lambda: "tester")

        runtime_dir = platform.get_runtime_dir()

        assert runtime_dir == tmp_path / "knowlin-tester"
        assert runtime_dir.stat().st_mode & 0o777 == 0o700

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="requires os.getuid")
    def test_runtime_dir_wrong_owner_raises(self, tmp_path, monkeypatch):
        from knowlin_mcp import platform

        monkeypatch.setattr(platform.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(platform.getpass, "getuser", lambda: "tester")

        runtime_dir = tmp_path / "knowlin-tester"
        runtime_dir.mkdir(mode=0o700)

        current_uid = os.getuid()
        fake_uid = current_uid + 1 if current_uid != 0 else 1
        monkeypatch.setattr(platform.os, "getuid", lambda: fake_uid)

        with pytest.raises(PermissionError):
            platform.get_runtime_dir()


class TestStructuredLogging:
    """Tests for logging configuration and levels."""

    def test_debug_log_emits_debug_record(self, caplog):
        from knowlin_mcp.utils import debug_log

        with caplog.at_level(logging.DEBUG, logger="knowlin"):
            debug_log("auth ok", category="server")

        assert any(
            record.levelno == logging.DEBUG and record.message == "[server] auth ok"
            for record in caplog.records
        )

    def test_check_idle_timeout_logs_warning(self, tmp_path, caplog):
        from knowlin_mcp.server import IDLE_TIMEOUT, KnowledgeServer

        (tmp_path / ".knowledge-db").mkdir()

        server = KnowledgeServer(str(tmp_path))
        server.last_activity = time.time() - IDLE_TIMEOUT - 1

        with caplog.at_level(logging.WARNING, logger="knowlin"):
            assert server.check_idle_timeout() is True

        assert any(
            record.levelno == logging.WARNING and "Idle timeout" in record.message
            for record in caplog.records
        )
