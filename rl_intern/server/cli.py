import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local rl-intern run server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("rl-intern-server is local-only; use --host 127.0.0.1")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "rl-intern-server requires server dependencies. Run `uv sync --extra server`."
        ) from exc
    uvicorn.run("rl_intern.server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
