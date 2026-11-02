"""The hosted service: a public web front end and job queue, and build workers that pull from it.

The web side accepts a `mapcreator.job.Job` and nothing else, queues it in
SQLite, and hands it to a worker that asks for work over HTTPS. Workers never
accept inbound connections. See `docs/architecture.md`.
"""
