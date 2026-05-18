try:
    from .cli import main
except ImportError:
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from layer_by_layer_method.cli import main


if __name__ == "__main__":
    main()
