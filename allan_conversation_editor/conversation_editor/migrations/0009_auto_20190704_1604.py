# Generated by Django 2.2.2 on 2019-07-04 13:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('conversation_editor', '0008_auto_20190704_1148'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chatline',
            name='parent',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='children', to='conversation_editor.ChatLine'),
        ),
    ]