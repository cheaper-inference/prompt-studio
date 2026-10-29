"""Offline checks for telemetry opt-in and Django's startup/request boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import sentry_init

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PROBE = """
import importlib.util, json
from django.conf import settings
from django.core.exceptions import DisallowedHost
from django.test import RequestFactory
from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.http import HttpResponse
spec = importlib.util.spec_from_file_location('audit_settings', 'webapp_django/optimizer_site/settings.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
settings.configure(**{k: v for k, v in vars(module).items() if k.isupper()})
# Populate an empty app registry for Django's CSRF error template without
# starting optimizer jobs or touching a database.
settings.INSTALLED_APPS = []
import django
django.setup()
factory = RequestFactory()
request = factory.get('/', HTTP_HOST='untrusted.invalid', HTTP_X_FORWARDED_PROTO='https')
try:
    request.get_host()
    rejects_unknown_host = False
except DisallowedHost:
    rejects_unknown_host = True
csrf = CsrfViewMiddleware(lambda request: HttpResponse())
request = factory.post('/', HTTP_HOST='localhost', HTTP_ORIGIN='https://foreign.invalid')
csrf.process_request(request)
request.META['HTTP_X_CSRFTOKEN'] = get_token(request)
response = csrf.process_view(request, lambda request: HttpResponse(), (), {})
def accepts_origin(origin):
    request = factory.post('/', secure=True, HTTP_HOST='localhost', HTTP_ORIGIN=origin)
    csrf.process_request(request)
    request.META['HTTP_X_CSRFTOKEN'] = get_token(request)
    request.COOKIES[settings.CSRF_COOKIE_NAME] = request.META['CSRF_COOKIE']
    return csrf.process_view(request, lambda request: HttpResponse(), (), {}) is None
print(json.dumps({
    'hosts': module.ALLOWED_HOSTS, 'origins': module.CSRF_TRUSTED_ORIGINS,
    'debug': module.DEBUG, 'secure_session': module.SESSION_COOKIE_SECURE,
    'secure_csrf': module.CSRF_COOKIE_SECURE, 'is_secure': factory.get('/', HTTP_X_FORWARDED_PROTO='https').is_secure(),
    'rejects_unknown_host': rejects_unknown_host,
    'rejects_foreign_origin': response is not None and response.status_code == 403,
    'accepts_same_origin': accepts_origin('https://localhost'),
    'accepts_configured_origin': accepts_origin(module.CSRF_TRUSTED_ORIGINS[0]) if module.CSRF_TRUSTED_ORIGINS else None,
}))
"""


class TelemetryTests(unittest.TestCase):
    def check_environment(self, env):
        sdk = Mock()
        sdk.is_initialized.return_value = False
        with patch.dict(os.environ, env, clear=True), patch.object(sentry_init, 'sentry_sdk', sdk):
            sentry_init.init_sentry()
        return sdk

    def test_production_without_a_dsn_sends_nothing(self):
        for value in (None, '', '  '):
            with self.subTest(dsn=value):
                env = {'RAILWAY_ENVIRONMENT_NAME': 'production'}
                if value is not None:
                    env['SENTRY_DSN'] = value
                self.check_environment(env).init.assert_not_called()

    def test_dsn_alone_does_not_enable_development_reporting(self):
        self.check_environment({'SENTRY_DSN': 'https://public@example.invalid/1'}).init.assert_not_called()

    def test_operator_dsn_enables_only_production_errors(self):
        for environment_var in ('SENTRY_ENVIRONMENT', 'RAILWAY_ENVIRONMENT_NAME'):
            with self.subTest(environment_var=environment_var):
                dsn = 'https://public@example.invalid/1'
                sdk = self.check_environment({'SENTRY_DSN': dsn, environment_var: 'production'})
                sdk.init.assert_called_once_with(
                    dsn=dsn, environment='production', traces_sample_rate=0, send_default_pii=False)

    def test_existing_client_is_not_reinitialised(self):
        sdk = Mock()
        sdk.is_initialized.return_value = True
        with patch.dict(os.environ, {'SENTRY_DSN': 'https://public@example.invalid/1',
                                     'SENTRY_ENVIRONMENT': 'production'}, clear=True), \
                patch.object(sentry_init, 'sentry_sdk', sdk):
            sentry_init.init_sentry()
        sdk.init.assert_not_called()


class DjangoConfigurationTests(unittest.TestCase):
    def probe(self, **values):
        # Keep only OS process essentials; never inherit real credentials/telemetry.
        env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'TMPDIR') if k in os.environ}
        env.update(DJANGO_SECRET_KEY=secrets.token_urlsafe(64))
        env.update(values)
        return subprocess.run([sys.executable, '-c', SETTINGS_PROBE], cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=30)

    def test_invalid_keys_fail_without_echoing_the_key(self):
        for key in ('', 'change-me', 'a' * 64, 'django-insecure-' + secrets.token_urlsafe(64)):
            with self.subTest(length=len(key)):
                result = self.probe(DJANGO_SECRET_KEY=key)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Set DJANGO_SECRET_KEY', result.stderr)
                if key:
                    self.assertNotIn(key, result.stderr)

    def test_defaults_reject_unknown_hosts_and_cross_origin_posts(self):
        result = self.probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['hosts'], ['localhost', '127.0.0.1', '[::1]'])
        self.assertEqual(data['origins'], [])
        self.assertFalse(data['debug'])
        self.assertFalse(data['is_secure'])
        for key in ('secure_session', 'secure_csrf', 'rejects_unknown_host', 'rejects_foreign_origin', 'accepts_same_origin'):
            self.assertTrue(data[key], key)

    def test_explicit_local_http_development(self):
        result = self.probe(DJANGO_DEBUG='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data['debug'])
        self.assertFalse(data['secure_session'])
        self.assertFalse(data['secure_csrf'])

    def test_explicit_deployment_hosts_origins_and_proxy(self):
        result = self.probe(DJANGO_ALLOWED_HOSTS='app.example.org, localhost',
                            RAILWAY_PUBLIC_DOMAIN='gepa.up.railway.app', DJANGO_TRUST_PROXY='1',
                            DJANGO_CSRF_TRUSTED_ORIGINS='https://client.example.org')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['hosts'], ['app.example.org', 'localhost', 'gepa.up.railway.app'])
        self.assertEqual(data['origins'], ['https://client.example.org'])
        self.assertTrue(data['is_secure'])
        self.assertTrue(data['accepts_configured_origin'])

    def test_unrestricted_hosts_fail_startup(self):
        result = self.probe(DJANGO_ALLOWED_HOSTS='localhost, *')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must name hosts explicitly', result.stderr)


if __name__ == '__main__':
    unittest.main()
