from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('optimizer', '0013_minime_review'),
    ]

    operations = [
        migrations.AddField(
            model_name='minimelabel',
            name='judged_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
