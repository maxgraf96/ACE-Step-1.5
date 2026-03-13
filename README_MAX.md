# README_MAX

Launch commands for this ACE-Step fork on the GPU server, with external access enabled via `0.0.0.0`.

## API server

From the repo root:

```bash
uv run acestep-api --host 0.0.0.0 --port 8001
```

Tmux version:

```bash
tmux new -d -s ace-api 'cd /home/ubuntu/ACE-Step-1.5 && uv run acestep-api --host 0.0.0.0 --port 8001'
```

## Simple UI

From the repo root:

```bash
uv run acestep-simple --server-name 0.0.0.0 --port 7861 --api-base-url http://127.0.0.1:8001
```

Tmux version:

```bash
tmux new -d -s ace-ui 'cd /home/ubuntu/ACE-Step-1.5 && uv run acestep-simple --server-name 0.0.0.0 --port 7861 --api-base-url http://127.0.0.1:8001'
```

## Notes

- Binding to `0.0.0.0` exposes the services on the server network interfaces.
- The UI still points its backend to `http://127.0.0.1:8001`, which is correct when both processes run on the same machine.
- Default external ports for this setup are `8001` for the API and `7861` for the UI.
