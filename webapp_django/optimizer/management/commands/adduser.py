"""Issue a Prompt Studio account manually (self-signup is off by default).

    python manage.py adduser alice                      # prompts for a password
    python manage.py adduser alice --password s3cr3t
    python manage.py adduser alice --password s3cr3t --update   # reset an existing password
"""
from __future__ import annotations

import getpass

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Create a Prompt Studio user (or reset a password with --update)."

    def add_arguments(self, parser):
        parser.add_argument("username")
        parser.add_argument("--password", default=None, help="if omitted, you're prompted")
        parser.add_argument("--update", action="store_true", help="reset password if the user exists")

    def handle(self, *args, **o):
        u = (o["username"] or "").strip()
        if not u:
            raise CommandError("username is required")
        pw = o["password"] or getpass.getpass("Password: ")
        if not pw:
            raise CommandError("password is required")
        user = User.objects.filter(username=u).first()
        if user and not o["update"]:
            raise CommandError(f"user {u!r} already exists (use --update to reset the password)")
        if user:
            user.set_password(pw); user.save()
            self.stdout.write(self.style.SUCCESS(f"Password reset for {u!r}"))
        else:
            User.objects.create_user(username=u, password=pw)
            self.stdout.write(self.style.SUCCESS(f"Created user {u!r}"))
