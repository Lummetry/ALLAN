# Generated by Django 2.2.2 on 2019-07-04 06:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('conversation_editor', '0006_chatline_human'),
    ]

    operations = [
        migrations.AddField(
            model_name='chat',
            name='status',
            field=models.PositiveSmallIntegerField(default=0),
        ),
    ]
