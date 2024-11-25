import hashlib
from django.http import JsonResponse
from .models import User
from .Serializer import UserSerializer, LoginSerializer
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils.decorators import method_decorator  # Import method_decorator
from django.views.decorators.csrf import csrf_exempt  # Import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

class UserList(ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class SignupView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'User created successfully!'}, status=status.HTTP_201_CREATED)
        print(serializer.errors)  # Log validation errors
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


class LoginView(APIView):
    def post(self, request):
        # Deserialize the incoming JSON data
        serializer = LoginSerializer(data=request.data)

        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            
            # Try to get the user by email
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return JsonResponse({'error': 'Invalid credentials'}, status=400)
            
            hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        
        # Now compare the hashed version of the password with the one stored in the DB
            if hashed_password == user.password:
                refresh = RefreshToken.for_user(user)
                access_token = str(refresh.access_token)
                refresh_token = str(refresh)

                # Return success with the username, access token, and refresh token
                return JsonResponse({
                    'message': 'Login successful',
                    'username': user.username,  # Include the username here
                    'access': access_token,
                    'refresh': refresh_token,
                }, status=200)
            else:
                # Return error if password does not match
                return JsonResponse({'error': 'Invalid credentials'}, status=400)

        return JsonResponse({'error': 'Bad request, invalid data'}, status=400)
class ProtectedView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Access the authenticated user's username
        username = request.user.username

        return Response({"message": f"{username}. You have access to this protected view."})