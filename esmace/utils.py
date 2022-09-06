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
