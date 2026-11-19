# Logprob API
The OpenRouter proxy for the logprob viewer. It adds the server key, turns logprobs on and caps the cost. It stores nothing.

## Where
| Item | Value |
|---|---|
| Web address | logprob.ideitas.online |
| Service (unit) | ideitas-logprob |
| Port | 8818 |
| Source | /opt/ideitas/logprob (API only). The page source is /opt/ideitas/logprob-gui (scan 2026-11-17) |
| Web root | /var/www/ideitas.online/logprob-gui |
| Data | /var/lib/ideitas/logprob |
| Env file (path only) | /var/lib/ideitas/logprob/logprob.env |

## Run
systemd unit `ideitas-logprob` runs as user `logprob`.
- ExecStart: `/opt/ideitas/logprob/venv/bin/python3 -m uvicorn app:app --host 127.0.0.1 --port 8818`
- Restart=always (the unit has Restart=always).
- Boot: /etc/ideitas-stagger.conf starts this unit at boot. The unit is not enabled by itself.
- Start or stop: use `systemctl` on the unit. Do not run it by hand.

## Build and deploy
This folder has only `app.py` and `venv`.
1. Edit `app.py`.
2. Run `systemctl restart ideitas-logprob`.
The page build steps: Not known yet. Check `/opt/ideitas/logprob-gui/web`.

## Data and backups
The data folder holds only env files. No database. No automatic backups. The owner makes backups only on request.

## Notes
- Owner note: the server key is shared. There is no BYOK.
- The page is a gold, static Devs-style screen with timelines (owner note).

Last checked: 2026-11-17
