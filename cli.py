"""
CLI Interactivo para EM-Predictor
Gestión de servicios, logs y despliegue con ngrok
"""
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
except ImportError:
    print("Instalando dependencias del CLI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

console = Console()
BASE_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

SERVICES = {
    "postgres": {"port": 5432, "docker": True, "desc": "Base de datos TimescaleDB"},
    "redis": {"port": 6379, "docker": True, "desc": "Cache y Feature Store"},
    "api_gateway": {"port": 8000, "docker": True, "desc": "API Gateway (FastAPI)"},
    "ml_inference": {"port": 8001, "docker": True, "desc": "Servicio de Inferencia ML"},
    "unified_app": {"port": 8080, "docker": False, "desc": "Aplicación Unificada"},
    "webapp": {"port": 5173, "docker": False, "desc": "Frontend React"},
}

NGROK_PORT = 8080


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: str, cwd: Optional[Path] = None, capture: bool = True) -> tuple[int, str]:
    """Ejecuta un comando y retorna código + salida."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or BASE_DIR,
            capture_output=capture,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def check_port(port: int) -> bool:
    """Verifica si un puerto está en uso."""
    code, _ = run_cmd(f"netstat -an | findstr :{port}")
    return code == 0


def get_docker_status() -> dict:
    """Obtiene estado de contenedores Docker."""
    code, output = run_cmd("docker-compose ps --format json")
    if code != 0:
        return {}
    
    status = {}
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                import json
                data = json.loads(line)
                name = data.get("Service", data.get("Name", ""))
                state = data.get("State", "unknown")
                status[name] = state
            except:
                pass
    return status


# ══════════════════════════════════════════════════════════════════════════════
# COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_status():
    """Muestra estado de todos los servicios."""
    console.print("\n[bold cyan]📊 Estado de Servicios[/]\n")
    
    docker_status = get_docker_status()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Servicio", style="cyan")
    table.add_column("Puerto")
    table.add_column("Estado")
    table.add_column("Descripción")
    
    for name, info in SERVICES.items():
        port = info["port"]
        port_active = check_port(port)
        
        if info["docker"]:
            state = docker_status.get(name, "stopped")
            if state == "running":
                status = "[green]● Running[/]"
            elif port_active:
                status = "[yellow]● Port Active[/]"
            else:
                status = "[red]○ Stopped[/]"
        else:
            status = "[green]● Running[/]" if port_active else "[dim]○ Not Started[/]"
        
        table.add_row(name, str(port), status, info["desc"])
    
    console.print(table)
    console.print()


def cmd_start(target: str = "all"):
    """Inicia servicios."""
    console.print(f"\n[bold green]🚀 Iniciando servicios: {target}[/]\n")
    
    if target in ("all", "infra", "backend"):
        console.print("[cyan]→ Levantando infraestructura Docker...[/]")
        code, output = run_cmd("docker-compose up -d postgres redis")
        if code == 0:
            console.print("[green]  ✓ PostgreSQL y Redis iniciados[/]")
        else:
            console.print(f"[red]  ✗ Error: {output}[/]")
            return
        
        time.sleep(2)
    
    if target in ("all", "backend"):
        console.print("[cyan]→ Levantando servicios de aplicación...[/]")
        code, output = run_cmd("docker-compose up -d api_gateway ml_inference")
        if code == 0:
            console.print("[green]  ✓ API Gateway y ML Inference iniciados[/]")
        else:
            console.print(f"[yellow]  ⚠ Algunos servicios no iniciaron: {output[:100]}[/]")
    
    if target in ("all", "unified"):
        console.print("[cyan]→ Iniciando aplicación unificada...[/]")
        unified_dir = BASE_DIR / "services" / "unified_app"
        if (unified_dir / "main.py").exists():
            # Iniciar en background
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"],
                cwd=unified_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            )
            console.print("[green]  ✓ Unified App iniciada en :8080[/]")
        else:
            console.print("[yellow]  ⚠ unified_app/main.py no encontrado[/]")
    
    if target in ("all", "frontend"):
        webapp_dir = BASE_DIR / "webapp"
        if webapp_dir.exists():
            console.print("[cyan]→ Iniciando frontend React...[/]")
            subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=webapp_dir,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            )
            console.print("[green]  ✓ Frontend iniciado en :5173[/]")
        else:
            console.print("[dim]  ○ Directorio webapp/ no existe aún[/]")
    
    console.print("\n[bold green]✓ Servicios iniciados[/]\n")
    cmd_status()


def cmd_stop():
    """Detiene todos los servicios."""
    console.print("\n[bold yellow]⏹ Deteniendo servicios...[/]\n")
    
    code, _ = run_cmd("docker-compose down")
    if code == 0:
        console.print("[green]  ✓ Contenedores Docker detenidos[/]")
    
    # Matar procesos en puertos conocidos
    for name, info in SERVICES.items():
        if not info["docker"] and check_port(info["port"]):
            run_cmd(f"taskkill /F /IM python.exe 2>nul", capture=False)
            run_cmd(f"taskkill /F /IM node.exe 2>nul", capture=False)
    
    console.print("[green]✓ Servicios detenidos[/]\n")


def cmd_logs(service: str = "all"):
    """Muestra logs de servicios."""
    console.print(f"\n[bold cyan]📜 Logs: {service}[/]\n")
    
    if service == "all":
        run_cmd("docker-compose logs --tail=50 -f", capture=False)
    else:
        run_cmd(f"docker-compose logs --tail=50 -f {service}", capture=False)


def cmd_ngrok():
    """Inicia túnel ngrok."""
    console.print("\n[bold magenta]🌐 Iniciando ngrok...[/]\n")
    
    if not check_port(NGROK_PORT):
        console.print(f"[yellow]⚠ Puerto {NGROK_PORT} no está activo. Inicia unified_app primero.[/]")
        if Confirm.ask("¿Iniciar unified_app ahora?"):
            cmd_start("unified")
            time.sleep(3)
    
    console.print(f"[cyan]→ Exponiendo puerto {NGROK_PORT}...[/]")
    console.print("[dim]Presiona Ctrl+C para detener[/]\n")
    
    subprocess.run(f"ngrok http {NGROK_PORT}", shell=True)


def cmd_metrics():
    """Muestra métricas del sistema."""
    console.print("\n[bold cyan]📈 Métricas del Sistema[/]\n")
    
    # Docker stats
    code, output = run_cmd("docker stats --no-stream --format \"table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\"")
    if code == 0 and output.strip():
        console.print(output)
    else:
        console.print("[dim]No hay contenedores corriendo[/]")
    
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def show_banner():
    """Muestra banner de bienvenida."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║          🧠 EM-Predictor CLI v1.0                         ║
    ║          Plataforma de Predicción de Esclerosis Múltiple  ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def interactive_menu():
    """Menú interactivo principal."""
    show_banner()
    
    while True:
        console.print("\n[bold]Opciones:[/]")
        console.print("  [cyan]1[/] → Iniciar todos los servicios")
        console.print("  [cyan]2[/] → Ver estado")
        console.print("  [cyan]3[/] → Detener servicios")
        console.print("  [cyan]4[/] → Ver logs")
        console.print("  [cyan]5[/] → Métricas")
        console.print("  [cyan]6[/] → Exponer con ngrok")
        console.print("  [cyan]7[/] → Inicio rápido (backend + frontend + ngrok)")
        console.print("  [cyan]q[/] → Salir")
        
        choice = Prompt.ask("\n[bold]Selecciona una opción[/]", default="1")
        
        if choice == "1":
            target = Prompt.ask(
                "¿Qué iniciar?",
                choices=["all", "infra", "backend", "unified", "frontend"],
                default="all"
            )
            cmd_start(target)
        elif choice == "2":
            cmd_status()
        elif choice == "3":
            cmd_stop()
        elif choice == "4":
            service = Prompt.ask("Servicio", default="all")
            cmd_logs(service)
        elif choice == "5":
            cmd_metrics()
        elif choice == "6":
            cmd_ngrok()
        elif choice == "7":
            cmd_start("all")
            console.print("\n[bold]Esperando 5s para que los servicios arranquen...[/]")
            time.sleep(5)
            cmd_ngrok()
        elif choice.lower() == "q":
            console.print("\n[bold green]¡Hasta luego! 👋[/]\n")
            break
        else:
            console.print("[red]Opción no válida[/]")


def main():
    """Punto de entrada."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EM-Predictor CLI")
    parser.add_argument("command", nargs="?", choices=["start", "stop", "status", "logs", "ngrok", "metrics"])
    parser.add_argument("--target", "-t", default="all", help="Target para start: all, infra, backend, unified, frontend")
    parser.add_argument("--service", "-s", default="all", help="Servicio para logs")
    
    args = parser.parse_args()
    
    if args.command is None:
        interactive_menu()
    elif args.command == "start":
        cmd_start(args.target)
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
    elif args.command == "logs":
        cmd_logs(args.service)
    elif args.command == "ngrok":
        cmd_ngrok()
    elif args.command == "metrics":
        cmd_metrics()


if __name__ == "__main__":
    main()
