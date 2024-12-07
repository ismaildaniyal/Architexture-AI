import hashlib
import random
from urllib import request
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from .models import User, UserAuth
from .Serializer import UserSerializer, LoginSerializer,validate_Email_Serializer,StorePasswordSerializer
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils.decorators import method_decorator  # Import method_decorator
from django.views.decorators.csrf import csrf_exempt  # Import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
import requests # type: ignore
from django.core.mail import send_mail
from django.conf import settings
from django.core.cache import cache
otp=0
from django.contrib.sites.shortcuts import get_current_site
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.utils.http import urlsafe_base64_decode
from .models import UserAuth, User
from django.db import IntegrityError
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
class UserList(ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


# class SignupView(APIView):
#     def post(self, request):
#         # Get the email from the request data
#         email = request.data.get('email')

#         # Call the email verification function before proceeding
#         email_valid = self.verify_email(email)

#         if not email_valid:
#             # If the email is invalid, return a 300 status with an error message
#             return Response({'error': 'Invalid email address'}, status=status.HTTP_300_MULTIPLE_CHOICES)

#         # Proceed with user creation if the email is valid
#         serializer = UserSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response({'message': 'User created successfully!'}, status=status.HTTP_201_CREATED)
#         else:
#             print(serializer.errors)  # Log validation errors
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     def verify_email(self, email):
#         """Function to verify email using Hunter.io API"""
#         api_key = 'a9cbd7926b9a73b87ddc4976825cc1b9059790c7'
#         try:
#             # Send a request to the Hunter.io API to verify the email
#             response = requests.get(f'https://api.hunter.io/v2/email-verifier?email={email}&api_key={api_key}')
#             data = response.json()
            
#             # Check if the email is deliverable
#             if data.get("data", {}).get("result") == "deliverable":
#                 return True
#             else:
#                 return False
#         except Exception as e:
#             print(f"Error verifying email: {e}")
#             return False
    


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
import traceback

class GenerateOtpView(APIView):
    def post(self, request):
        serializer = validate_Email_Serializer(data=request.data)
        print("hi")
        if serializer.is_valid():
            email = serializer.validated_data['email']

            # Check if email exists in the database
            if not self.is_email_exist(email):
                return Response({'error': 'Account does not exist. Please sign up.'}, status=status.HTTP_404_NOT_FOUND)
            
            otp = random.randint(1000, 9999)
            otp_str = str(otp)

            # Send the OTP to the provided email
            try:
                send_mail(
                    'Your OTP Code',
                    f'Your OTP code is {otp_str}',
                    settings.EMAIL_HOST_USER,  # Email from settings.py
                    [email],
                    fail_silently=False,
                )
                # Store OTP in cache with a 10-second expiration
                cache.set(email, otp_str, timeout=60)  # OTP stored for 10 seconds
                return Response({"message": "OTP sent to your email address"}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_404_NOT_FOUND)

    def is_email_exist(self, email):
        """Check if the email exists in the database."""
        try:
            user = User.objects.get(email=email)
            return True
        except User.DoesNotExist:
            return False


class StorePasswordView(APIView):
    def post(self, request):
        serializer = StorePasswordSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Password updated successfully."}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VerifyOtpView(APIView):
    def post(self, request):
        # Get the email and OTP from the request data
        print("Request Data:", request.data)
        email = request.data.get('email')
        otp = request.data.get('otp')

        # Check if the email and OTP are provided
        if not email or not otp:
            return Response({"error": "Email and OTP are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Log incoming request for debugging purposes (optional)
        print(f"Received email: {email}, OTP: {otp}")

        # Retrieve the OTP stored in cache for the provided email
        cached_otp = cache.get(email)

        # Check if the OTP exists and is valid
        if not cached_otp:
            return Response({"error": "OTP has expired or was not generated. Please request a new one."}, status=status.HTTP_408_REQUEST_TIMEOUT)

        # Validate the OTP
        if otp == cached_otp:
            # OTP is valid, return success message
            return Response({"message": "OTP verified successfully!"}, status=status.HTTP_200_OK)
        else:
            # OTP is incorrect
            return Response({"error": "Invalid OTP. Please try again."}, status=status.HTTP_406_NOT_ACCEPTABLE)


class SignupView1(APIView):
    def post(self, request):
        email = request.data.get('email')
        
        # Check if email already exists in the database
        if User.objects.filter(email=email).exists():
            return Response({'error': 'A user with this email already exists.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Verify email before proceeding
        email_valid = self.verify_email(email)
        
        if not email_valid:
            return Response({'error': 'Invalid email address'}, status=status.HTTP_300_MULTIPLE_CHOICES)

        # Store user data temporarily in UserAuth model
        user_auth = UserAuth(
            username=request.data.get('username'),
            email=email,
            password=hashlib.sha256(request.data.get('password').encode('utf-8')).hexdigest()
        )
        user_auth.save()
        
        # Generate the verification token and send the email
        user_auth.generate_verification_token()
        self.send_verification_email(user_auth, request)

        return Response({'message': 'User created. Please check your email for verification link.'}, 
                         status=status.HTTP_201_CREATED)

    def verify_email(self, email):
        """Function to verify email using Hunter.io API"""
        api_key = 'a9cbd7926b9a73b87ddc4976825cc1b9059790c7'
        try:
            response = requests.get(f'https://api.hunter.io/v2/email-verifier?email={email}&api_key={api_key}')
            data = response.json()

            if data.get("data", {}).get("result") == "deliverable":
                return True
            else:
                return False
        except Exception as e:
            print(f"Error verifying email: {e}")
            return False

    def send_verification_email(self, user_auth, request):
        """Send the email verification link to the user."""
        uid = urlsafe_base64_encode(str(user_auth.id).encode())
        verification_url = f"{get_current_site(request).domain}{reverse('verify_email', kwargs={'uidb64': uid, 'token': user_auth.verification_token})}"
        
        subject = "Verify Your Email"
        message = f"Click the link to verify your email: {verification_url}"
        
        send_mail(subject, message, "noreply@yourdomain.com", [user_auth.email])

class VerifyEmailView(APIView):
    def get(self, request, uidb64, token):
        try:
            # Decode the user ID from the URL
            user_auth_id = urlsafe_base64_decode(uidb64).decode()
            user_auth = get_object_or_404(UserAuth, id=user_auth_id)

            # Check if the token matches and is still valid
            if user_auth.verification_token == token and user_auth.is_token_valid():
                # Move the data from UserAuth to User and delete the temporary record
                user = User.objects.create(
                    username=user_auth.username,
                    email=user_auth.email,
                    password=user_auth.password,
                )
                # Delete the temporary UserAuth record
                user_auth.delete()
                # Redirect to the login page after successful verification
                return HttpResponseRedirect("http://localhost:3000/login")
            else:
                return HttpResponse("The verification link has expired or is invalid. Please sign up again.")
        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")