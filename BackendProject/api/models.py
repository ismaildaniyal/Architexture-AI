from django.db import models

# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=45, null=False)  # Unique and not null
    email = models.EmailField(primary_key=True, null=False)  # Email as the primary key
    password = models.CharField(max_length=256, null=False)  # Not null