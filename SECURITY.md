# Security

mapcreator includes a public web service: a map page, a job API and queue, and build workers that fetch data and write tracks people download and install. Security reports are welcome.

## Reporting a problem

Please don't open a public issue. Report it privately instead: on this repository's **Security** tab, choose **Report a vulnerability**. That's GitHub's private vulnerability reporting, so only the maintainers see the report, and we can work on a fix together there.

Include:

- which part is affected (the web app, the worker, the build pipeline, a built track, or the container examples);
- how to reproduce it;
- what someone could do with it.

This is a personal project, so a reply may take a few days. Please give us a reasonable time to fix the problem before you publish it.

## In scope

- The web app and job API (`src/mapcreator/service/`), including the rate limits, the Turnstile check and the worker token.
- The build worker and pipeline. Job input is meant to be coordinates, enums and booleans only, so anything else in a job that reaches a shell, a file parser or the network is a bug.
- Built tracks: a file in a track that could harm a player's PC or Content Manager.
- The container and proxy examples in `deploy/examples/`.

Not in scope: the in-game test harness (`harness/`), which is for development only, and the third-party data services the pipeline reads from.

## Supported versions

Only the latest code on `main`. Fixes aren't backported.
