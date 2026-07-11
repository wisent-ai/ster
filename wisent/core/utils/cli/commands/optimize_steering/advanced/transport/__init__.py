"""Transport-based steering optimization: PRZELOM config and RL loop."""
from wisent.core.utils.cli.optimize_steering.transport.method_configs_transport import PrzelomConfig


def __getattr__(name):
    if name == "execute_transport_rl":
        from wisent.core.utils.cli.optimize_steering.transport.transport_rl import execute_transport_rl
        return execute_transport_rl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PrzelomConfig", "execute_transport_rl"]
