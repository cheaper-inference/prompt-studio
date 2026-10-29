from django.db import migrations, models
from django.db.models import BigIntegerField, Case, Sum, TextField, Value, When
from django.db.models.functions import Cast, Coalesce, Greatest, Length


def account_existing_receipts(apps, schema_editor):
    calls = apps.get_model("optimizer", "InferenceCall").objects.using(schema_editor.connection.alias)
    jobs = apps.get_model("optimizer", "JobState").objects.using(schema_editor.connection.alias)
    # Freeze historical accounting here rather than importing runtime limits.
    # Four bytes per character is a conservative UTF-8 bound on every backend;
    # calculate in SQL without loading historical response bodies into memory.
    identifiers = sum((Length(field) for field in
                       ("model", "request_id", "call_key", "request_hash", "phase", "purpose")), Value(0))
    actual = 4 * (Coalesce(Length(Cast("response", TextField())), Value(4))
                  + Coalesce(Length(Cast("usage", TextField())), Value(2)) + identifiers) + 4096
    calls.update(storage_bytes=Case(
        When(status="complete", then=actual),
        default=Greatest(actual, Value(4 * 1024 * 1024 + 4096) + 4 * identifiers),
        output_field=BigIntegerField(),
    ))
    for row in calls.order_by().values("job_id").annotate(total=Sum("storage_bytes")).iterator(chunk_size=100):
        jobs.filter(pk=row["job_id"]).update(receipt_storage_bytes=row["total"])


class Migration(migrations.Migration):
    dependencies = [("optimizer", "0007_interrupt_prequeue_jobs")]
    operations = [
        migrations.AddField(model_name="jobstate", name="receipt_storage_bytes",
                            field=models.PositiveBigIntegerField(default=0)),
        migrations.AddField(model_name="inferencecall", name="storage_bytes",
                            field=models.PositiveBigIntegerField(default=0)),
        migrations.RunPython(account_existing_receipts, migrations.RunPython.noop),
    ]
