from django.apps import AppConfig

class OptimizerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "optimizer"
    # Job recovery belongs to the durable worker lease, never web-process startup.
