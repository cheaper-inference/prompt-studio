"""What the web app sends back must not carry an attacker's destination or a server-side error text."""
import os
from unittest.mock import patch
from urllib.parse import urlencode

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings

from llm_endpoint import DEFAULT_KEY_VAR, DEFAULT_URL_VAR, EndpointConfigError

PASSWORD = 'correct horse battery staple'


# The default hasher is deliberately slow; these tests only need a password that verifies.
@override_settings(PASSWORD_HASHERS=['django.contrib.auth.hashers.MD5PasswordHasher'])
class LoginRedirectTests(TestCase):
    def setUp(self):
        User.objects.create_user(username='alice', password=PASSWORD)

    def _location_after_login(self, target=None):
        query = '' if target is None else '?' + urlencode({'next': target})
        response = Client().post('/login/' + query, {'username': 'alice', 'password': PASSWORD})
        self.assertEqual(response.status_code, 302)
        return response['Location']

    def test_a_path_on_this_site_is_followed(self):
        for target in ('/', '/?job=abc', '/api/me'):
            with self.subTest(target=target):
                self.assertEqual(self._location_after_login(target), target)

    def test_another_site_falls_back_to_the_index(self):
        for target in ('https://evil.example/', 'http://evil.example', '//evil.example/path',
                       '/\\evil.example', 'https:evil.example'):
            with self.subTest(target=target):
                self.assertEqual(self._location_after_login(target), '/')

    def test_no_destination_goes_to_the_index(self):
        self.assertEqual(self._location_after_login(), '/')


class HealthDetailTests(TestCase):
    def setUp(self):
        self.client.force_login(User.objects.create_user(username='alice'))

    def test_missing_variables_are_named_without_the_exception_text(self):
        absent = {name: '' for name in (DEFAULT_KEY_VAR, DEFAULT_URL_VAR)}
        with patch.dict(os.environ, absent), \
                patch('llm_endpoint.configure', side_effect=EndpointConfigError('No API key.')):
            body = self.client.get('/api/health').json()
        self.assertFalse(body['endpoint_available'])
        self.assertEqual(body['missing'], [DEFAULT_KEY_VAR, DEFAULT_URL_VAR])
        self.assertIn(DEFAULT_KEY_VAR, body['detail'])
        self.assertIn(DEFAULT_URL_VAR, body['detail'])
        self.assertNotIn('EndpointConfigError', body['detail'])

    def test_an_unexpected_failure_is_not_echoed(self):
        present = {DEFAULT_KEY_VAR: 'fixture-key', DEFAULT_URL_VAR: 'http://127.0.0.1:1234/v1'}
        failure = PermissionError(13, 'Permission denied', '/srv/prompt-studio/.env')
        with patch.dict(os.environ, present), patch('llm_endpoint.configure', side_effect=failure), \
                self.assertLogs('optimizer.services', 'WARNING'):
            body = self.client.get('/api/health').json()
        self.assertFalse(body['endpoint_available'])
        self.assertEqual(body['missing'], [])
        self.assertTrue(body['detail'])
        for leaked in ('/srv/prompt-studio', 'PermissionError', 'Permission denied'):
            self.assertNotIn(leaked, body['detail'])
