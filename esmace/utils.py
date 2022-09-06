from platform import python_version_tuple
from typing import Dict, Sequence

from sklearn.exceptions import NotFittedError


def check_is_fitted(value, msg=None):
    if msg is None:
        msg = 'The instance is not fitted: '

    if hasattr(value, "__is_fitted__"):
        fitted = value.__is_fitted__()
    else:
        fitted = [
            v for v in vars(value) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(value).__name__})


if int(python_version_tuple()[1]) >= 10:
    from dataclasses import dataclass
else:
    from dataclasses import dataclass as dataclass_
    def dataclass(*args: Sequence, **kwargs: Dict):
        if 'slots' in kwargs:
            kwargs.pop('slots')

        return dataclass_(*args, **kwargs)
