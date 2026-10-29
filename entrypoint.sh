#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Applying database migrations..."
python manage.py migrate

# Execute the main container command (passed from CMD in Dockerfile)
exec "$@"
