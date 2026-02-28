# dub

AI video dubbing pipeline.

## Local Development

Start Redis:

```bash
docker compose up -d redis
```

Start the API and worker in separate terminals:

```bash
poe api
poe worker
```

## Docker

```bash
docker compose up
```