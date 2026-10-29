#!/usr/bin/env python
"""Django management entry point for the prompt-optimizer web app."""
import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "optimizer_site.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
