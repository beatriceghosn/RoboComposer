#!/usr/bin/env python3
"""Serve the RoboComposer UI from ./frontend (default http://127.0.0.1:8765)."""

from __future__ import annotations

import http.server
import os
import socketserver
import webbrowser


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    root = os.path.join(os.path.dirname(__file__), "frontend")
    os.chdir(root)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Serving {root}")
        print(f"Open {url}")
        try:
            webbrowser.open(url)
        except OSError:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
