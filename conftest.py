import sys

def pytest_runtest_setup(item):
    """Unload safeserialize before each test, allowing us to detect dependency
    problems.

    In the test modules, safeserialize should not be imported at the module
    level. Each test function should import (from safeserialize) what it needs.

    """

    # Remove any safeserialize-related globals. There shouldn't be any, but
    # just in case ...
    for var in ("reader,writer,Serializer,serializer,"
                "dump,dumps,load,loads".split(",")):
        try:
            delattr(item.module, var)
            print("deleted", var)
        except Exception as exc:
            pass

    # Unload safeserialize (and its submodules).
    for module in sys.modules.copy():
        if module.startswith("safeserialize"):
            del sys.modules[module]
