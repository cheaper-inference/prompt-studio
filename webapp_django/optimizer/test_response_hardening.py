"""What the web app sends back must not carry an attacker's destination or a server-side error text."""
from urllib.parse import urlencode

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings

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

