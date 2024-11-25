import hashlib
from rest_framework import serializers
from .models import User
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


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        # Use `create_user` to handle password hashing
        validated_data['password'] = hashlib.sha256(validated_data['password'].encode('utf-8')).hexdigest()
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
        )
        return user
    
    
class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
