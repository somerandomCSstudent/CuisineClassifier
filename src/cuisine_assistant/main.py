from pathlib import Path

from cuisine_assistant.bootstrap import build_application


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    build_application(project_root).run()


if __name__ == "__main__":
    main()
