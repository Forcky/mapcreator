"""Stage runners: wire providers to core maths. Each stage reads and writes the work folder."""

# The stages of a build, in order. The CLI runs them; the worker reports which one it's on, and the web
# app shows that on the job page. Kept here, free of imports, so the web app needn't load the pipeline.
STAGE_NAMES = ("route", "profile", "section", "mesh", "scenery", "track", "validate")
