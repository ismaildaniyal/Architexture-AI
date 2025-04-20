from django.db import models
import hashlib
from django.utils import timezone
from datetime import timedelta

class UserAuth(models.Model):
    username = models.CharField(max_length=45, null=False)
    email = models.EmailField(unique=True, null=False)
    password = models.CharField(max_length=256, null=False)
    verification_token = models.CharField(max_length=256)
    token_created_at = models.DateTimeField(auto_now_add=True)
    email_verified = models.BooleanField(default=False)

    def generate_verification_token(self):
        """Generate a token for email verification."""
        self.verification_token = hashlib.sha256(str(self.id).encode()).hexdigest()
        self.save()

    def is_token_valid(self):
        """Check if the token is still valid (within 5 minutes)."""
        expiration_time = self.token_created_at + timedelta(minutes=5)
        return timezone.now() < expiration_time
# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=45, null=False) 
    email = models.EmailField(primary_key=True, null=False)  
    password = models.CharField(max_length=256, null=False)  


class Chat(models.Model):
    chat_id = models.IntegerField()  
    user = models.ForeignKey(User, on_delete=models.CASCADE, to_field='email')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['chat_id', 'user'], name='unique_chat')  # Composite key (chat_id, email)
        ]

    def __str__(self):
        return f"Chat {self.chat_id} by {self.user.email}"




# ChatPrompt model (api_chatprompt)
class ChatPrompt(models.Model):
    id = models.AutoField(primary_key=True)
    prompt_text = models.TextField()
    output_text = models.TextField(blank=True, null=True)
    image_path = models.CharField(max_length=255, blank=True, null=True)
    design_path = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)  # No "to_field" here

    def __str__(self):
        return f"Prompt {self.id} for Chat {self.chat.chat_id}"
