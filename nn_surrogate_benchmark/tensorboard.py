import socket
import subprocess
import time
import webbrowser


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def ensure_tensorboard_running(logdir: str, port: int = 6006) -> None:
    if not is_port_in_use(port):
        print(f"Starting TensorBoard on port {port}...")
        subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)
        webbrowser.open(f"http://localhost:{port}")
    else:
        print(f"TensorBoard already running on port {port}")
