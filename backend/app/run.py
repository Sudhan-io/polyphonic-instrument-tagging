import os
import sys
import logging
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

for path in [PROJECT_ROOT, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import uvicorn
from backend.app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audiotag.server")


async def main():
    raw_port = os.environ.get("PORT", "10000")
    try:
        env_port = int(raw_port)
    except ValueError:
        env_port = 10000

    # Bind to both Render's default 10000 and 8000 (and any custom $PORT)
    target_ports = sorted(list({env_port, 8000, 10000}))
    logger.info("Starting AudioTag AI production server on target ports: %s", target_ports)

    servers = []
    tasks = []

    for port in target_ports:
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=30,
        )
        server = uvicorn.Server(config)
        servers.append(server)
        tasks.append(asyncio.create_task(server.serve()))
        logger.info("Registered Uvicorn listener on 0.0.0.0:%d", port)

    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutdown signal received, closing all listeners.")
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
