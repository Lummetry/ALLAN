from django.db import models
from django.contrib.auth.models import User

# Create your models here.
from django.utils.timezone import now


class Domain(models.Model):
    domainName = models.CharField(max_length=50, blank=False, null=False)

    def __str__(self):
        return '{}'.format(self.domainName)


class Chat(models.Model):
    title = models.CharField(max_length=100, blank=False, null=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    created_user = models.ForeignKey(User, editable=False, null=True, blank=True, on_delete=models.CASCADE)
    domain = models.ForeignKey(Domain, on_delete=models.CASCADE)
    status = models.PositiveSmallIntegerField(default=0)


class Label(models.Model):
    name = models.CharField(max_length=100, blank=False, null=False)
    domain = models.ForeignKey(Domain, on_delete=models.CASCADE)
    chat = models.ManyToManyField(Chat)


class ChatLine(models.Model):
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    message = models.CharField(max_length=1024, blank=False, null=False)
    parent = models.ForeignKey('self', null=True, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    human = models.BooleanField(default=True, blank=False, null=False)
    created_user = models.ForeignKey(User, editable=False, null=True, blank=True, on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
