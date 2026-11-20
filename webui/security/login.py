from http.cookies import SimpleCookie
from urllib.parse import parse_qs
import secrets
import time

from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

CONFIG = {
    "cookie_name": "mydets_session",
    "session_seconds": 30 * 24 * 3600,
    "max_body_bytes": 1024
}

LOGIN_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MyDets</title>
<style>
body { font-family: system-ui, sans-serif; background: #f4f5f7; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
form { background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0, 0, 0, .1); width: 340px; }
h1 { margin: 0 0 8px; font-size: 20px; }
p { margin: 0 0 16px; color: #555; font-size: 14px; }
input, button { width: 100%; box-sizing: border-box; padding: 10px; font-size: 15px; border-radius: 6px; }
input { border: 1px solid #bbb; margin-bottom: 12px; }
button { border: 0; background: #1565c0; color: #fff; cursor: pointer; }
.error { color: #c62828; margin-bottom: 12px; font-size: 14px; }
</style>
</head>
<body>
<form method="post" action="/login">
<h1>MyDets</h1>
<p>Paste the access token printed in the terminal where the server runs.</p>
$error
<input type="password" name="token" autocomplete="off" autofocus required>
<button type="submit">Log in</button>
</form>
</body>
</html>
"""


class TokenGate:
    """ASGI middleware that lets nothing through until the access token of the server was entered on its login page."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        """Initialize the gate.

        Args:
            app: application to protect.
            token: access token that opens a session.
        """
        self.app = app
        self.token = token
        self.sessions: dict[str, float] = {}

    def is_logged_in(self, scope: Scope) -> bool:
        """Check the session cookie of a request, and forget the sessions that expired."""
        now = time.time()
        self.sessions = {session: expires for session, expires in self.sessions.items() if expires > now}
        headers = dict(scope["headers"])
        cookie = SimpleCookie(headers.get(b"cookie", b"").decode("latin-1"))
        morsel = cookie.get(CONFIG["cookie_name"])
        return morsel is not None and any(secrets.compare_digest(morsel.value, session) for session in self.sessions)

    @staticmethod
    def is_https(scope: Scope) -> bool:
        """Check whether the browser used https, directly or through the proxy of a platform."""
        forwarded = dict(scope["headers"]).get(b"x-forwarded-proto", b"").decode()
        return scope["scheme"] in ("https", "wss") or forwarded == "https"

    async def login(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Show the login page, and open a session when the token on it is right."""
        request = Request(scope, receive)
        error = ""

        if request.method == "POST" and request.headers.get("content-length", "0").isdigit() and int(request.headers["content-length"]) <= CONFIG["max_body_bytes"]:
            entered = parse_qs((await request.body()).decode()).get("token", [""])[0]

            if secrets.compare_digest(entered.encode(), self.token.encode()):
                session = secrets.token_urlsafe(32)
                self.sessions[session] = time.time() + CONFIG["session_seconds"]
                response = RedirectResponse("/", status_code=303)
                response.set_cookie(CONFIG["cookie_name"], session, max_age=CONFIG["session_seconds"], httponly=True, samesite="strict", secure=self.is_https(scope))
                await response(scope, receive, send)
                return

            error = '<div class="error">Wrong token.</div>'

        response = HTMLResponse(LOGIN_PAGE.replace("$error", error), status_code=401 if error else 200)
        response.headers.update({"Cache-Control": "no-store", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"})
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or self.is_logged_in(scope):
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.close", "code": 1008})
        elif scope["path"] == "/login":
            await self.login(scope, receive, send)
        else:
            response: Response = RedirectResponse("/login", status_code=303)
            await response(scope, receive, send)
