"""PRO / pay-what-you-want supporter licenses + PayPal donation helpers.

The orchestrator itself is and stays free (MIT, open source). "PRO" is a
*supporter* tier on the honour system: when someone donates whatever they want
via PayPal, an optional, self-hosted webhook issues a signed supporter license
and emails it to their PayPal address. The license unlocks a supporter badge and
our gratitude — not core features (you can't meaningfully paywall open source).

Everything here is standard library only:
  * `issue_license` / `verify_license` / `decode_license` - HMAC-signed keys.
  * `handle_paypal_event` - turn a PayPal webhook event into a license + email.
  * `serve` - a tiny webhook server (PayPal -> license -> SMTP email).
  * `cli_handle` - the `ofo --donate / --pro / --activate` commands.

The live loop needs YOUR config (host it yourself):
  OFO_LICENSE_SECRET   - secret used to sign licenses (keep private!)
  OFO_SMTP_HOST/PORT/USER/PASS/FROM - to actually send the license email
  OFO_PAYPAL_SHARED_SECRET (optional) - a shared token checked on the webhook
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import smtplib
import sys
import time
from email.message import EmailMessage

DONATE_URL = "https://www.paypal.com/donate/?hosted_button_id=N4NJ5QYSY3GLC"
PRODUCT = "Free-Tier AI Orchestrator"
_PREFIX = "OFO-PRO."
_ACCEPTED_EVENTS = {
    "PAYMENT.CAPTURE.COMPLETED", "PAYMENT.SALE.COMPLETED",
    "CHECKOUT.ORDER.APPROVED", "CHECKOUT.ORDER.COMPLETED",
}


# --------------------------------------------------------------------------- #
# License keys (HMAC-signed, offline-decodable)
# --------------------------------------------------------------------------- #
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _secret(secret=None) -> bytes:
    secret = secret if secret is not None else os.environ.get("OFO_LICENSE_SECRET", "")
    if not secret:
        raise RuntimeError("OFO_LICENSE_SECRET is not set (needed to sign/verify licenses).")
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def issue_license(email: str, tier: str = "pro", days: int = None,
                  amount=None, secret=None) -> str:
    issued = int(time.time())
    payload = {"e": email, "t": tier, "i": issued,
               "x": (issued + days * 86400) if days else 0}
    if amount is not None:
        payload["a"] = str(amount)
    token = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    sig = hmac.new(_secret(secret), token.encode("ascii"), hashlib.sha256).hexdigest()[:20]
    return "%s%s.%s" % (_PREFIX, token, sig)


def decode_license(key: str):
    """Parse a key's payload WITHOUT verifying the signature (honour-system read)."""
    try:
        if not key.startswith(_PREFIX):
            return None
        body = key[len(_PREFIX):]
        token, _sig = body.split(".", 1)
        return json.loads(_b64d(token))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def verify_license(key: str, secret=None) -> bool:
    try:
        if not key.startswith(_PREFIX):
            return False
        token, sig = key[len(_PREFIX):].split(".", 1)
        expect = hmac.new(_secret(secret), token.encode("ascii"),
                          hashlib.sha256).hexdigest()[:20]
        if not hmac.compare_digest(sig, expect):
            return False
        payload = json.loads(_b64d(token))
        return not payload.get("x") or payload["x"] >= time.time()
    except (ValueError, TypeError, RuntimeError, json.JSONDecodeError):
        return False


# --------------------------------------------------------------------------- #
# Local activation (stored alongside the CLI config)
# --------------------------------------------------------------------------- #
def license_path() -> str:
    from .config import config_home
    return os.path.join(config_home(), "license.key")


def store_license(key: str) -> str:
    path = license_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(key.strip() + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_local_license():
    try:
        with open(license_path(), "r", encoding="utf-8") as fh:
            key = fh.read().strip()
        return key, decode_license(key)
    except OSError:
        return None, None


# --------------------------------------------------------------------------- #
# PayPal webhook -> license -> email
# --------------------------------------------------------------------------- #
def _dig_email(event: dict) -> str:
    res = event.get("resource", {}) or {}
    payer = res.get("payer", {}) or {}
    for path in (payer.get("email_address"),
                 (payer.get("payer_info", {}) or {}).get("email"),
                 res.get("payer_email"),
                 (event.get("payer", {}) or {}).get("email_address")):
        if path:
            return str(path)
    return ""


def _dig_amount(event: dict):
    res = event.get("resource", {}) or {}
    amt = res.get("amount", {}) or {}
    val = amt.get("value") or res.get("amount_value") or amt.get("total")
    cur = amt.get("currency_code") or amt.get("currency") or ""
    return ("%s %s" % (val, cur)).strip() if val else None


def render_email(email: str, key: str, amount=None) -> "tuple[str, str]":
    subject = "Your %s PRO supporter license 💛" % PRODUCT
    thanks = ("for your %s donation" % amount) if amount else "for your donation"
    body = (
        "Hi,\n\n"
        "Thank you %s to %s — it genuinely helps. 💛\n\n"
        "Here is your pay-what-you-want PRO supporter license:\n\n"
        "    %s\n\n"
        "Activate it with:\n"
        "    ofo --activate %s\n\n"
        "It marks you as a PRO supporter. The tool stays fully free and open "
        "source either way.\n\n"
        "— The %s project\n%s\n"
        % (thanks, PRODUCT, key, key, PRODUCT, DONATE_URL)
    )
    return subject, body


def _smtp_send(to_addr: str, subject: str, body: str) -> bool:
    host = os.environ.get("OFO_SMTP_HOST", "")
    if not host:
        return False
    port = int(os.environ.get("OFO_SMTP_PORT", "587"))
    user = os.environ.get("OFO_SMTP_USER", "")
    pwd = os.environ.get("OFO_SMTP_PASS", "")
    sender = os.environ.get("OFO_SMTP_FROM", user or "no-reply@localhost")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        try:
            smtp.starttls()
            smtp.ehlo()
        except smtplib.SMTPException:
            pass
        if user:
            smtp.login(user, pwd)
        smtp.send_message(msg)
    return True


def handle_paypal_event(event: dict, secret=None, send: bool = True) -> dict:
    """Turn a PayPal webhook event into a license (+ optional email)."""
    etype = event.get("event_type", "")
    if etype and etype not in _ACCEPTED_EVENTS:
        return {"skipped": True, "reason": "ignored event_type %s" % etype}
    email = _dig_email(event)
    if not email:
        return {"skipped": True, "reason": "no payer email in event"}
    amount = _dig_amount(event)
    key = issue_license(email, tier="pro", amount=amount, secret=secret)
    subject, body = render_email(email, key, amount)
    sent = False
    if send:
        try:
            sent = _smtp_send(email, subject, body)
        except Exception as exc:  # noqa: BLE001 - never crash the webhook
            return {"email": email, "amount": amount, "key": key,
                    "subject": subject, "body": body, "sent": False,
                    "error": str(exc)}
    return {"email": email, "amount": amount, "key": key,
            "subject": subject, "body": body, "sent": sent}


def serve(host: str = "127.0.0.1", port: int = 8770) -> int:
    """Run the PayPal webhook -> license-email server (stdlib only)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    shared = os.environ.get("OFO_PAYPAL_SHARED_SECRET", "")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            return

        def _reply(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._reply(200, {"ok": True, "service": "ofo-pro license server"})

        def do_POST(self):
            if shared and self.headers.get("X-OFO-Secret", "") != shared:
                self._reply(403, {"ok": False, "error": "bad shared secret"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                event = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError):
                self._reply(400, {"ok": False, "error": "invalid json"})
                return
            try:
                result = handle_paypal_event(event, send=True)
            except RuntimeError as exc:   # e.g. missing OFO_LICENSE_SECRET
                self._reply(500, {"ok": False, "error": str(exc)})
                return
            self._reply(200, {"ok": True, "issued": bool(result.get("key")),
                              "emailed": bool(result.get("sent"))})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print("PRO license webhook listening on http://%s:%d/  (POST PayPal events)"
          % (host, port))
    print("Point your PayPal webhook at this URL (behind HTTPS in production).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# CLI surface (used by `ofo --donate/--pro/--activate` and `python -m ...pro`)
# --------------------------------------------------------------------------- #
def cli_handle(args, ui) -> int:
    if getattr(args, "activate", None):
        key = args.activate.strip()
        payload = decode_license(key)
        if not payload:
            ui.error("That doesn't look like a valid license key.")
            return 2
        store_license(key)
        ui.note("PRO supporter license activated for %s. Thank you! 💛"
                % payload.get("e", "you"))
        return 0
    if getattr(args, "pro", False):
        key, payload = load_local_license()
        if payload:
            ui.note("PRO supporter ✓  (%s) — thank you for supporting development! 💛"
                    % payload.get("e", ""))
        else:
            ui.note("Free edition. Everything works — PRO is just a supporter badge.")
        ui.note("Pay what you want: " + DONATE_URL)
        ui.note("Got a key by email? Activate it with:  ofo --activate <key>")
        return 0
    # default: --donate
    ui.note("Support %s (pay what you want):" % PRODUCT)
    ui.note("  " + DONATE_URL)
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="ofo-pro", description="PRO supporter license tools.")
    sub = p.add_subparsers(dest="cmd")
    pi = sub.add_parser("issue", help="issue a signed license (needs OFO_LICENSE_SECRET)")
    pi.add_argument("--email", required=True)
    pi.add_argument("--tier", default="pro")
    pi.add_argument("--days", type=int, default=None)
    pv = sub.add_parser("verify", help="verify a license key")
    pv.add_argument("key")
    pd = sub.add_parser("decode", help="decode (no verify) a license key")
    pd.add_argument("key")
    ps = sub.add_parser("serve", help="run the PayPal webhook -> email server")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8770)
    sub.add_parser("selftest", help="run offline self-tests")
    args = p.parse_args(argv)

    if args.cmd == "issue":
        print(issue_license(args.email, args.tier, args.days))
        return 0
    if args.cmd == "verify":
        ok = verify_license(args.key)
        print("VALID" if ok else "INVALID")
        return 0 if ok else 1
    if args.cmd == "decode":
        print(json.dumps(decode_license(args.key), indent=2))
        return 0
    if args.cmd == "serve":
        return serve(args.host, args.port)
    if args.cmd == "selftest":
        return 0 if run_self_tests() else 1
    p.print_help()
    return 0


def run_self_tests() -> bool:
    secret = "test-secret-123"
    key = issue_license("fan@example.com", "pro", secret=secret)
    assert verify_license(key, secret=secret), "fresh key should verify"
    assert decode_license(key)["e"] == "fan@example.com"
    assert not verify_license(key + "x", secret=secret), "tampered key must fail"
    assert not verify_license(key, secret="other"), "wrong secret must fail"
    expired = issue_license("old@example.com", days=-1, secret=secret)
    assert not verify_license(expired, secret=secret), "expired key must fail"

    event = {"event_type": "PAYMENT.CAPTURE.COMPLETED",
             "resource": {"amount": {"value": "7.00", "currency_code": "EUR"},
                          "payer": {"email_address": "donor@example.com"}}}
    res = handle_paypal_event(event, secret=secret, send=False)
    assert res["email"] == "donor@example.com", res
    assert verify_license(res["key"], secret=secret), "issued key must verify"
    assert res["key"] in res["body"] and "donor@example.com" in res["email"]
    ignored = handle_paypal_event({"event_type": "BILLING.SUBSCRIPTION.CREATED"},
                                  secret=secret, send=False)
    assert ignored.get("skipped"), "non-payment events are ignored"
    return True


if __name__ == "__main__":
    sys.exit(main())
