import torch
from functools import wraps

def dynamic_custom_fwd(cast_inputs=None):
    """
    Custom decorator that dynamically determines the device_type from input arguments
    and applies torch.amp.custom_fwd.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            device_type = 'cpu'  # Default to CPU
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    device_type = arg.device.type
                    break
                elif isinstance(arg, torch.nn.Module):
                    device_type = get_device(arg).type
                    break

            @torch.amp.custom_fwd(device_type=device_type, cast_inputs=cast_inputs)
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func(*args, **kwargs)
        return wrapper
    return decorator


def get_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        raise RuntimeError("The model has no parameters. Cannot determine device.")