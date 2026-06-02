"""
Configura una tarea en Windows Task Scheduler para actualizar los datos diariamente.
Ejecutar una sola vez como administrador.

Uso:
  python setup_scheduler.py           # programa actualización a las 7:00 AM
  python setup_scheduler.py --time 06:00
  python setup_scheduler.py --remove  # elimina la tarea
"""

import os
import sys
import argparse
import subprocess

TASK_NAME = "TennisPredictor_DailyUpdate"


def setup(run_time: str = "07:00"):
    python = sys.executable
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "update_data.py")
    project_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = (
        f'schtasks /create /tn "{TASK_NAME}" '
        f'/tr "\\"{python}\\" \\"{script}\\"" '
        f'/sc daily /st {run_time} '
        f'/sd {__import__("datetime").date.today().strftime("%d/%m/%Y")} '
        f'/f'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Tarea creada: '{TASK_NAME}'")
        print(f"  Horario: todos los días a las {run_time}")
        print(f"  Script : {script}")
        print(f"\nVerifica en: Administrador de tareas > Biblioteca del programador de tareas")
    else:
        print(f"Error al crear la tarea:\n{result.stderr}")
        print("\nAlternativa — ejecutar manualmente cada día:")
        print(f"  python src/update_data.py")


def remove():
    cmd = f'schtasks /delete /tn "{TASK_NAME}" /f'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Tarea '{TASK_NAME}' eliminada.")
    else:
        print(f"No se encontró la tarea o error:\n{result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="07:00", help="Hora de ejecución HH:MM (default: 07:00)")
    parser.add_argument("--remove", action="store_true", help="Eliminar la tarea programada")
    args = parser.parse_args()

    if args.remove:
        remove()
    else:
        setup(args.time)
