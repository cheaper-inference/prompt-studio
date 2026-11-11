# Shared UI foundations

Prompt Studio uses selected design tokens, typography rules, and Keak brand
vectors from the company's shared UI foundations, version 0.0.106. The Django
adapter retains native controls and the existing optimization workflows; there
is no React dependency, Node build, private npm package, or token needed to run it.
The selected company-owned files shipped here are covered by the repository’s
[MIT license](../LICENSE); this does not distribute or relicense the entire private kit.

The public distribution uses system fonts. It does not bundle Switzer, a font
loader, or any other font files, and it does not fetch fonts from a CDN. The
company-hosted application may therefore have slightly different typography.

## Running locally

Use development mode to serve source assets without collecting them first:

```bash
DJANGO_DEBUG=1 python webapp_django/manage.py runserver 127.0.0.1:8000
```

Supply a generated `DJANGO_SECRET_KEY` and follow the account/database setup in
[the web app guide](../webapp_django/README.md). Keep debug mode off on public
deployments. Run the job worker with the same environment and working directory
to process optimization jobs.

## Production assets

WhiteNoise serves CSS, JavaScript, and SVGs inside the existing Django/Gunicorn
process. The Docker build collects, fingerprints, and compresses the assets;
no separate asset server or runtime process is required. For a non-Docker install,
run this before starting Gunicorn (and before running the Django tests):

```bash
python webapp_django/manage.py collectstatic --noinput
```

`DJANGO_STATIC_ROOT` overrides the default `webapp_django/staticfiles` output
directory. Keep databases, credentials, and uploads outside this directory.
Collected assets are generated files and are excluded from Git and Docker input.
