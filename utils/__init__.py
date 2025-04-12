_MODULES = {}


def return_none():
    return None


def import_modules_into_registry():
    print("Importing modules into registry")
    import algos


def register_module(category=None, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        local_category = category
        if local_category is None:
            local_category = cls.__name__ if name is None else name

        # Create category (if does not exist)
        if local_category not in _MODULES:
            _MODULES[local_category] = {}

        # Add module to the category
        local_name = cls.__name__ if name is None else name
        if name in _MODULES[local_category]:
            raise ValueError(
                f"Already registered module with name: {local_name} in category: {category}"
            )

        _MODULES[local_category][local_name] = cls
        return cls

    return _register


def get_module(category, name):
    module = _MODULES.get(category, dict()).get(name, None)
    if module is None:
        raise ValueError(f"No module named `{name}` found in category: `{category}`")
    return module
