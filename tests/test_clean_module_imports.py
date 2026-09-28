"""
Unit test verifying that every module across data/, models/, training/, and evaluation/
can be cleanly imported in an isolated Python subprocess without NameError or ImportError.
Guarantees missing typing/runtime imports cannot hide behind pytest test-runner caching.
"""

import os
import sys
import subprocess
import pytest


def _get_python_modules():
    """Discovers all python module dot-paths in data, models, training, evaluation."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    modules = []

    target_dirs = ['data', 'models', 'training', 'evaluation']
    for t_dir in target_dirs:
        dir_path = os.path.join(base_dir, t_dir)
        if not os.path.isdir(dir_path):
            continue
        for f in os.listdir(dir_path):
            if f.endswith('.py') and not f.startswith('__'):
                mod_name = f"{t_dir}.{f[:-3]}"
                modules.append(mod_name)
    return modules


@pytest.mark.parametrize("module_name", _get_python_modules())
def test_clean_subprocess_import(module_name):
    """Executes `import <module_name>` and resolves all type annotations in an isolated clean Python subprocess."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env = os.environ.copy()
    env["PYTHONPATH"] = base_dir

    py_code = (
        f"import importlib, typing\n"
        f"mod = importlib.import_module('{module_name}')\n"
        f"for attr_name in dir(mod):\n"
        f"    obj = getattr(mod, attr_name)\n"
        f"    if callable(obj) and getattr(obj, '__module__', '') == '{module_name}':\n"
        f"        try:\n"
        f"            typing.get_type_hints(obj)\n"
        f"        except NameError as e:\n"
        f"            raise NameError(f'Missing type annotation import in {{obj.__qualname__}}: {{e}}') from e\n"
    )

    cmd = [sys.executable, "-c", py_code]
    res = subprocess.run(
        cmd,
        cwd=base_dir,
        env=env,
        capture_output=True,
        text=True
    )

    assert res.returncode == 0, (
        f"Failed clean subprocess import/annotation check for '{module_name}':\n"
        f"STDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}"
    )
