"""
Email Service — sends transactional emails via SMTP.

Used for password reset OTP codes. Supports Gmail App Passwords,
SendGrid SMTP, or any standard SMTP provider.

Configuration via environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM_EMAIL, SMTP_FROM_NAME
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


def _is_configured() -> bool:
    """Check if SMTP settings are configured."""
    return bool(
        _settings.SMTP_HOST
        and _settings.SMTP_USER
        and _settings.SMTP_PASSWORD
    )


def send_otp_email(to_email: str, otp_code: str, username: str = "") -> bool:
    """
    Send a password reset OTP code via email.

    Returns True on success, False on failure.
    """
    if not _is_configured():
        logger.error("SMTP not configured — cannot send OTP email")
        return False

    from_email = _settings.SMTP_FROM_EMAIL or _settings.SMTP_USER
    from_name = _settings.SMTP_FROM_NAME or "Portfolio Tracker"

    subject = f"{otp_code} is your password reset code"

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
        .container {{ max-width: 480px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .header {{ background: #1a73e8; padding: 32px 24px; text-align: center; }}
        .header h1 {{ color: #ffffff; margin: 0; font-size: 22px; font-weight: 600; }}
        .body {{ padding: 32px 24px; }}
        .greeting {{ font-size: 16px; color: #333; margin-bottom: 16px; }}
        .message {{ font-size: 15px; color: #555; line-height: 1.6; margin-bottom: 24px; }}
        .otp-box {{ background: #f0f4ff; border: 2px dashed #1a73e8; border-radius: 10px; padding: 20px; text-align: center; margin: 24px 0; }}
        .otp-code {{ font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #1a73e8; font-family: 'Courier New', monospace; }}
        .warning {{ font-size: 13px; color: #888; margin-top: 24px; line-height: 1.5; }}
        .footer {{ background: #fafafa; padding: 16px 24px; text-align: center; border-top: 1px solid #eee; }}
        .footer p {{ font-size: 12px; color: #999; margin: 0; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>📊 Portfolio Tracker</h1>
        </div>
        <div class="body">
          <p class="greeting">Hi{(' ' + username) if username else ''},</p>
          <p class="message">
            We received a request to reset your password. Use the code below to
            verify your identity:
          </p>
          <div class="otp-box">
            <div class="otp-code">{otp_code}</div>
          </div>
          <p class="message">
            This code expires in <strong>10 minutes</strong>. If you didn't
            request a password reset, you can safely ignore this email.
          </p>
          <p class="warning">
            ⚠ Never share this code with anyone. Our team will never ask for it.
          </p>
        </div>
        <div class="footer">
          <p>Portfolio Tracker &mdash; Secure Password Reset</p>
        </div>
      </div>
    </body>
    </html>
    """

    text_body = (
        f"Hi{(' ' + username) if username else ''},\n\n"
        f"Your password reset code is: {otp_code}\n\n"
        f"This code expires in 10 minutes.\n"
        f"If you didn't request this, ignore this email.\n\n"
        f"— Portfolio Tracker"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        port = _settings.SMTP_PORT
        use_tls = _settings.SMTP_USE_TLS

        if use_tls:
            server = smtplib.SMTP(_settings.SMTP_HOST, port, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()
        else:
            server = smtplib.SMTP_SSL(_settings.SMTP_HOST, port, timeout=15)

        server.login(_settings.SMTP_USER, _settings.SMTP_PASSWORD)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()

        logger.info("OTP email sent to %s", to_email)
        return True

    except Exception as exc:
        logger.error("Failed to send OTP email to %s: %s", to_email, exc)
        return False
