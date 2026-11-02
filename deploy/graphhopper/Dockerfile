# Route previews for the map page: GraphHopper with a car profile.
# Build from this folder:  docker build -t mapcreator-router:local .
# The graph is imported once (see README.md) and mounted read-only at /data.
FROM eclipse-temurin:21-jre

ARG GH_VERSION=11.0
# SHA-256 of graphhopper-web-11.0.jar from the project's GitHub release.
ARG GH_SHA256=b59c024afe172ec6ec85b6327006c3138ec58c7d0bcd26253d0e42853f613def

WORKDIR /gh
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o graphhopper.jar \
      "https://github.com/graphhopper/graphhopper/releases/download/${GH_VERSION}/graphhopper-web-${GH_VERSION}.jar" \
 && echo "${GH_SHA256}  graphhopper.jar" | sha256sum -c - \
 && apt-get purge -y curl && apt-get autoremove -y

COPY config.yml mapcreator_car.json ./
RUN useradd -r -u 10001 gh
USER gh
EXPOSE 8989
# `server` serves an existing graph; run `import` (see README.md) to build one.
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar /gh/graphhopper.jar \"${GH_COMMAND:-server}\" /gh/config.yml"]
