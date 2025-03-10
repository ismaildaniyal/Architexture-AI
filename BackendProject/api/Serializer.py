import hashlib
from rest_framework import serializers
from .models import User
import bcrypt
# from django.contrib.auth.hashers import make_password
# from django.contrib.auth import get_user_model, authenticate


# user_model = get_user_model()

# class UserRegisterSerializers(serializers.ModelSerializer):
#     class Meta:
#         model = user_model
#         fields = '__all__'
#         def create (self, clean_data):
#             user_obj = user_model.objects.create_user(email=clean_data['email'],
#                                                 password=clean_data['password'])
#             user_obj.username=clean_data['username']
#             user_obj.save
#             return user_obj


# class UserLoginSerializers(serializers.Serializer):
#     email = serializers.EmailField()
#     password = serializers.CharField()
#     def validate(self, clean_data):
#         user = authenticate(email=clean_data['email'], password=clean_data['password'])
#         if not user:
#             raise serializers.ValidationError('Invalid credentials')
#         return user
from .models import Chat, ChatPrompt

class ChatPromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatPrompt
        fields = ['id', 'prompt_text', 'output_text', 'image_path', 'created_at', 'chat']

class ChatSerializer(serializers.ModelSerializer):
    prompts = ChatPromptSerializer(many=True, read_only=True, source='chatprompt_set')

    class Meta:
        model = Chat
        fields = ['chat_id', 'created_at', 'prompts']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        # Use `create_user` to handle password hashing
        # validated_data['password'] = hashlib.sha256(validated_data['password'].encode('utf-8')).hexdigest()
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email'],
            password=bcrypt.hashpw(validated_data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')  # Hash and decode as string,
        )
        return user
    
    
class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)



class validate_Email_Serializer(serializers.Serializer):
    email = serializers.EmailField()

    def validate_email(self, value):
        """
        Check if the email exists in the database.
        """
        value = value.strip()
        # Log the email for debugging
        print(f"Validating email: {value}")
        
        try:
            from .models import User
            # Use a case-insensitive query
            user = User.objects.get(email__iexact=value)
            print(f"Email Data: {user}")
        except User.DoesNotExist:

            raise serializers.ValidationError("Account does not exist. Please sign up.")
        
        return value


class StorePasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=2)

    def validate(self, data):
        email = data.get("email")
        password = data.get("password")

        # Check if the user exists
        try:
            User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError("User with this email does not exist.")

        return data

    def save(self):
        email = self.validated_data["email"]
        password = self.validated_data["password"]
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # Get the user object and update the password
        user = User.objects.get(email=email)
        user.password = hashed_password.decode('utf-8')  # Hash the password
        user.save()

        return user