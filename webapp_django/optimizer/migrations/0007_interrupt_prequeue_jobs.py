from django.db import migrations


def interrupt_prequeue_jobs(apps, schema_editor):
    JobState = apps.get_model("optimizer", "JobState")
    # Old daemon threads cannot survive a deployment; no credentials/checkpoints
    # exist to resume them safely. Do not leave historical rows polling forever.
    JobState.objects.filter(status="running", lease_token__isnull=True).update(
        status="interrupted", stage="Interrupted", error="Job predates durable queue execution")


class Migration(migrations.Migration):
    dependencies = [("optimizer", "0006_inference_purposes")]
    operations = [migrations.RunPython(interrupt_prequeue_jobs, migrations.RunPython.noop)]
