# App TODOs

- Audit SQLite access pragma stuff — `immutable=1` in `HarvestDB` causes "database disk image is malformed" errors when the app reads a harvest DB mid-write (WAL not yet checkpointed). Investigate whether to check for WAL file existence, use normal locking mode, or add another safeguard. See `spd/harvest/db.py:79`.
