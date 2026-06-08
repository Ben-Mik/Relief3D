# Relief3D

Photogrammetry stack using OpenMVG and OpenMVS in addition to 3D-Annotator.

## Build

```bash
git clone https://github.com/Ben-Mik/Relief3D.git
cd Relief3D
cp .env.example .env   # fill in your values
docker compose build
docker compose up -d
```

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Default | Description |
|---|---|---|
| `DEPLOYMENT_DOMAIN` | — | Domain traefik routes on |
| `RELIEF3D_DATA_HOST` | `./data` | Host path for persistent data (jobs, meshes) |
| `RELIEF3D_SWEEP_HOURS` | `8` | Hours before uploads/meshes are auto-cleaned |

The container exposes Relief3D on `/relief3d` via traefik and talks to the annotator over the shared `traefik` Docker network.
