# Security Policy

## Supported Versions

We actively provide security updates for the following versions of **Cine Expert**:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0.0 | :x:                |

---

## Reporting a Vulnerability

We take the security and integrity of **Cine Expert** seriously. If you discover or suspect a security vulnerability, please do **NOT** open a public issue.

### How to Report
Please disclose the vulnerability responsibly by creating a private security advisory on GitHub or by contacting the maintainers directly.

When submitting a vulnerability report, please provide:
1. **Description**: Clear description of the vulnerability and its potential impact.
2. **Steps to Reproduce**: Detailed proof of concept (PoC) or minimal reproduction steps.
3. **Affected Components**: Specific endpoints, functions, or dependencies involved.
4. **Proposed Fix**: Any suggested mitigations or patches (if available).

### Response Timeline
- **Initial Response**: We aim to acknowledge vulnerability reports within **48 hours**.
- **Assessment & Fix**: We will assess the severity, collaborate with you on a resolution, and issue a patched release as soon as reasonably possible.
- **Public Disclosure**: Coordinated disclosure will occur once a fix has been deployed to ensure users can safely upgrade.

---

## Security Best Practices for Deployments

1. **TMDB API Key Protection**:
   - Store your `TMDB_API_KEY` inside `.env` or as a private environment variable on Vercel.
   - Never commit `.env` or raw secrets to version control.
2. **Input Validation**:
   - Keep user input sanitized when querying external services.
3. **Dependency Updates**:
   - Regularly update Python dependencies listed in `requirements.txt` to receive critical upstream security patches.
