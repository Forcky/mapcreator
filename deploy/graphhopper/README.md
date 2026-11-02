# Route-preview router (GraphHopper, car profile)

The map page's route previews come from this. Builds don't use it: the worker routes on its own OSM index, and the page sends points along the preview so both follow the same road.

`mapcreator_car.json` adds a mild preference for bigger roads on top of GraphHopper's built-in `car.json`. It matches the class weights the build's own router uses (`providers/osm/graph.py`), so previews and builds agree.

## Build the image

```sh
docker build -t mapcreator-router:local deploy/graphhopper
```

## Import a graph (once, then after each extract refresh)

Import on any machine with about 3 GB of RAM free, then copy `graph-car/` to the router host. For Australia this takes over an hour. Changing `config.yml` or `mapcreator_car.json` means importing again.

```sh
docker run --rm -e GH_COMMAND=import -e JAVA_OPTS="-Xmx2600m" --memory 3g \
  -v <data dir>:/data mapcreator-router:local
```

`<data dir>` must contain `osm/australia-latest.osm.pbf`. The graph is written to `<data dir>/graph-car/`.

## Serve

```sh
docker run -d --read-only -e JAVA_OPTS="-Xmx700m" --memory 1g -v <data dir>:/data mapcreator-router:local
```

The data mount stays writable: GraphHopper keeps a lock file beside the graph. Keep the router on a network only the web app can reach. It needs no internet access and publishes no ports. See `deploy/examples/compose.web.yaml`.
