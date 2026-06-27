# PRO supporter licenses (pay what you want)

The orchestrator is **free and MIT-licensed** — every feature works without
paying. "PRO" is an optional **supporter tier** on the honour system: people who
donate whatever they want via PayPal can receive a signed *supporter license*
(a badge + our thanks), optionally emailed to them automatically.

> Honest scope: because this is open source, PRO does **not** paywall features.
> The license is a thank-you token. The auto-email loop is something **you host**;
> it needs your PayPal webhook + an SMTP mailbox.

```
Donor pays (PayPal, pay-what-you-want)
        │  PayPal webhook  →  POST /paypal/webhook
        ▼
 orchestrator.pro server  ──issue signed license──►  email it via SMTP
        │
        ▼
 Donor: ofo --activate <key>     →  "PRO supporter ✓"
```

## 1. Try it offline (no setup)

```bash
python3 -m orchestrator.pro selftest          # offline self-tests
export OFO_LICENSE_SECRET="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
KEY=$(python3 -m orchestrator.pro issue --email fan@example.com)
echo "$KEY"
python3 -m orchestrator.pro verify "$KEY"      # -> VALID
ofo --activate "$KEY"                          # store it
ofo --pro                                       # show supporter status
```

## 2. Configure the live service (you host it)

Set these environment variables on your server:

| Variable | Purpose |
|----------|---------|
| `OFO_LICENSE_SECRET` | **Secret** used to sign licenses. Keep it private; never commit it. |
| `OFO_SMTP_HOST` / `OFO_SMTP_PORT` | Your SMTP server (e.g. `smtp.gmail.com` / `587`). |
| `OFO_SMTP_USER` / `OFO_SMTP_PASS` | SMTP login (use an app password). |
| `OFO_SMTP_FROM` | From address for the license email. |
| `OFO_PAYPAL_SHARED_SECRET` | *(optional)* token checked as `X-OFO-Secret` on the webhook. |

Run the webhook server:

```bash
python3 -m orchestrator.pro serve --host 0.0.0.0 --port 8770
```

Put it behind HTTPS (a reverse proxy such as Caddy/nginx) and register that
public URL as a PayPal webhook for the *payment completed* events
(`PAYMENT.CAPTURE.COMPLETED`, `PAYMENT.SALE.COMPLETED`). On each donation the
server extracts the payer's email + amount, issues a license, and emails it.

## 3. Production hardening (recommended)

- **Verify PayPal signatures.** This template trusts the POST body (optionally
  gated by `OFO_PAYPAL_SHARED_SECRET`). For real money, verify PayPal's
  `transmission_sig` against their certificate, or call PayPal's
  *verify-webhook-signature* API before issuing a license.
- **Rotate `OFO_LICENSE_SECRET`** carefully — changing it invalidates old keys.
- Keep a log/db of issued licenses if you want to revoke or re-send them.

That's it — pay-what-you-want support, with an automatic thank-you license.
