# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a security-sensitive finding. Use [GitHub private vulnerability reporting](https://github.com/KIM3310/memoryflow-lab/security/advisories/new) so remediation can be coordinated before disclosure.

Include reproduction steps, affected versions or commits, potential impact, and a suggested remediation when available.

## Supported scope

Security support applies to the default branch and the latest documented local or container execution path. Experimental measurement scripts and hardware-specific examples are best-effort unless the README states otherwise.

## Secret handling

Never commit API keys, access tokens, certificates, private keys, cookies, `.env` files, private datasets, or production logs. Rotate any credential that may have been exposed in a commit, artifact, issue, or screenshot.
