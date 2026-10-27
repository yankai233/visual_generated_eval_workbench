from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from src.web.server import create_app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图像生成可视化评估工作台")
    parser.add_argument("--server-name", default="0.0.0.0", help="监听地址")
    parser.add_argument("--server-port", type=int, default=7861, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    return parser.parse_args()


def discover_server_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "<server-ip>"


if __name__ == "__main__":
    args = parse_args()
    display_host = discover_server_ip() if args.server_name == "0.0.0.0" else args.server_name
    print(f"工作台地址: http://{display_host}:{args.server_port}")
    if args.server_name == "0.0.0.0":
        print("提示: 服务将暴露到可达网络，请按需配置内网、防火墙或访问控制。")
    uvicorn.run(
        "src.main:app",
        host=args.server_name,
        port=args.server_port,
        reload=args.reload,
    )

