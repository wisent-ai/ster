"""Re-enabled CLI command handlers.

These commands are registered in the argparse parser and documented, but
their dispatch handlers had been dropped from _COMMAND_MAP. Each module here
exposes an execute_<name>(args) handler wired back into cli/__init__.py
_IMPORT_MAP and main.py _COMMAND_MAP.
"""
