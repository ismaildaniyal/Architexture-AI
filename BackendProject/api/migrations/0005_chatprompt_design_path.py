# Generated by Django 5.1.3 on 2025-03-11 20:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_chat_chatprompt_chat_unique_chat'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatprompt',
            name='design_path',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
