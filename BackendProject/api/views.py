import uuid
from django.utils import timezone
import hashlib
import random
from urllib import request
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from .models import User, UserAuth, Chat, ChatPrompt
from .Serializer import ChatPromptSerializer, UserSerializer, LoginSerializer,validate_Email_Serializer,StorePasswordSerializer
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
import numpy as np
import torch
from Model_Implimentaion.test import validate_and_enhance_house_plan  # Import functions from file1
from Model_Implimentaion.model_implement import main  # Import model from file2
from Model_Implimentaion.Dplot import plot_3d_house_plan
import os
import bcrypt
class UserList(ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer




def process_input(user_input):
    if "code" in user_input.lower():
        return False
    else:
        return True
    
    
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django.utils import timezone
import uuid
import os
from .models import User, Chat, ChatPrompt
# from api.utils import process_input, validate_and_enhance_house_plan, main  # Import helper functions


def get_image(request, image_name):
    image_path = os.path.join(r"C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\images", image_name)
    print(f"Attempting to load image from: {image_path}")  # Debugging

    if not os.path.exists(image_path):
        print("Image not found!")  # Debugging
        return HttpResponse("Image not found", status=404)
    
    try:
        with open(image_path, "rb") as image_file:
            return HttpResponse(image_file.read(), content_type="image/png")
    except FileNotFoundError:
        return HttpResponse("Image not found", status=404)
def get_image_3D(request, image_name):
    image_path = os.path.join(r"C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\3D_plot", image_name)
    print(f"Attempting to load image from: {image_path}")  # Debugging

    if not os.path.exists(image_path):
        print("Image not found!")  # Debugging
        return HttpResponse("Image not found", status=404)
    
    try:
        with open(image_path, "rb") as image_file:
            return HttpResponse(image_file.read(), content_type="image/png")
    except FileNotFoundError:
        return HttpResponse("Image not found", status=404)




class HousePlanAPI(APIView):
    def post(self, request):
        user_input = request.data.get("input")
        prompt_id = request.data.get("promptId")
        email = request.data.get("email")

        # Check required fields
        missing_fields = [field for field in ["input", "promptId", "email"] if not request.data.get(field)]
        if missing_fields:
            return Response({"error": f"Missing required fields: {', '.join(missing_fields)}"}, status=status.HTTP_400_BAD_REQUEST)

        if not process_input(user_input):
            return Response({"error": "I'm not able to write any code."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get user by email
            user = User.objects.get(email=email)

            # Get or create chat
            chat, created = Chat.objects.get_or_create(
                chat_id=prompt_id, user=user, defaults={"created_at": timezone.now()}
            )
            chat_created_at = chat.created_at if not created else timezone.now()

            # Validate house plan
            validation_result = validate_and_enhance_house_plan(user_input)
            if not validation_result["is_valid"]:
                ChatPrompt.objects.create(
                    chat=chat, prompt_text=user_input, output_text=validation_result["reason"], 
                    image_path=None,  design_path=None, created_at=chat_created_at
                )
                return Response({"error": validation_result["reason"]}, status=status.HTTP_400_BAD_REQUEST)

            # Generate an image and save it
            # image_id = uuid.uuid4().hex  # Generate unique image ID
            # image_folder = "C:\Users\SMART TECH\Documents\FYP\BackendProject\Images"
            # os.makedirs(image_folder, exist_ok=True)  # Ensure directory exists
            # image_path = os.path.join(image_folder, f"final_image_{image_id}.png")
            image_path=main()  # Call main function with image path

            # Save in the database
            chat_prompt = ChatPrompt.objects.create(
                chat=chat, prompt_text=user_input, output_text="Success", 
                image_path=image_path, design_path=None, created_at=chat_created_at
            )

            # Read image file and return in response
            with open(image_path, "rb") as image_file:
                return HttpResponse(image_file.read(), content_type="image/png")

        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            error_message = str(e)
            if 'chat' in locals():
                ChatPrompt.objects.create(
                    chat=chat, prompt_text=user_input, output_text=error_message, 
                    image_path=None,design_path=None, created_at=chat_created_at
                )
            return Response({"error": error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HousePlanAPI1(APIView):
    def post(self, request):
        user_input = request.data.get("input")
        prompt_id = request.data.get("promptId")
        email = request.data.get("email")

        # Check required fields
        missing_fields = [field for field in ["input", "promptId", "email"] if not request.data.get(field)]
        if missing_fields:
            return Response({"error": f"Missing required fields: {', '.join(missing_fields)}"}, status=status.HTTP_400_BAD_REQUEST)

        if not process_input(user_input):
            return Response({"error": "I'm not able to write any code."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get user by email
            user = User.objects.get(email=email)

            # Get or create chat
            chat, created = Chat.objects.get_or_create(
                chat_id=prompt_id, user=user, defaults={"created_at": timezone.now()}
            )
            chat_created_at = chat.created_at if not created else timezone.now()

            # Validate house plan
            validation_result = validate_and_enhance_house_plan(user_input)
            if not validation_result["is_valid"]:
                ChatPrompt.objects.create(
                    chat=chat, prompt_text=user_input, output_text=validation_result["reason"], 
                    image_path=None, design_path=None, created_at=chat_created_at
                )
                return Response({"error": validation_result["reason"]}, status=status.HTTP_400_BAD_REQUEST)

            # Generate an image and save it
            # image_id = uuid.uuid4().hex  # Generate unique image ID
            # image_folder = "C:\Users\SMART TECH\Documents\FYP\BackendProject\Images"
            # os.makedirs(image_folder, exist_ok=True)  # Ensure directory exists
            # image_path = os.path.join(image_folder, f"final_image_{image_id}.png")
            image_path=main()  # Call main function with image path
            Design_path= plot_3d_house_plan()
            # Save in the database
            chat_prompt = ChatPrompt.objects.create(
                chat=chat, prompt_text=user_input, output_text="Success", 
                image_path=Design_path, design_path=image_path, created_at=chat_created_at
            )
            # Read image file and return in response
            with open(Design_path, "rb") as image_file:
                return HttpResponse(image_file.read(), content_type="image/png")

        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            error_message = str(e)
            if 'chat' in locals():
                ChatPrompt.objects.create(
                    chat=chat, prompt_text=user_input, output_text=error_message, 
                    image_path=None, design_path=None, created_at=chat_created_at
                )
            return Response({"error": error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserChatAPIView(APIView):
    def get(self, request):
        email = request.GET.get('email','').strip()
        chat_id = request.GET.get('chat_id')
        if not email:  # If email is empty or missing
            return Response({"error": "Email parameter is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate user existence
        user = get_object_or_404(User, email__iexact=email.strip())

        # If chat_id is provided, fetch all prompts for that chat
        if chat_id:
            chat = get_object_or_404(Chat, chat_id=chat_id, user=user)
            prompts = ChatPrompt.objects.filter(chat=chat).order_by('created_at')
            return Response(ChatPromptSerializer(prompts, many=True).data, status=status.HTTP_200_OK)

        # If chat_id is NOT provided, fetch all chats and the latest chat's prompts
        chats = Chat.objects.filter(user=user).order_by('-created_at')
        latest_chat = chats.first()
        latest_chat_prompts = ChatPrompt.objects.filter(chat=latest_chat).order_by('-created_at') if latest_chat else []

        # Prepare response
        response_data = {
            'all_chats': [{'chat_id': chat.chat_id, 'created_at': chat.created_at} for chat in chats],
            'latest_chat': {
                'chat_id': latest_chat.chat_id if latest_chat else None,
                'created_at': latest_chat.created_at if latest_chat else None,
                'prompts': ChatPromptSerializer(latest_chat_prompts, many=True).data
            } if latest_chat else None
        }

        return Response(response_data, status=status.HTTP_200_OK)
class DeleteUserChatAPIView(APIView):
    def delete(self, request):
        email = request.GET.get('email')
        chat_id = request.GET.get('chat_id')

        # Validate user
        user = get_object_or_404(User, email=email)

        # Validate chat
        chat = get_object_or_404(Chat, chat_id=chat_id, user=user)

        # Delete all related chat prompts
        ChatPrompt.objects.filter(chat=chat).delete()

        # Delete the chat itself
        chat.delete()

        return Response({'message': 'Chat and related prompts deleted successfully'}, status=status.HTTP_200_OK)

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
    


@method_decorator(csrf_exempt, name='dispatch')
class LoginView(APIView):
    def post(self, request):
        try:
            serializer = LoginSerializer(data=request.data)
            if serializer.is_valid():
                email = serializer.validated_data['email']
                password = serializer.validated_data['password']
                
                try:
                    user = User.objects.get(email=email)
                except User.DoesNotExist:
                    return JsonResponse({'error': 'Invalid credentials'}, status=400)
                
                if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                    refresh = RefreshToken()
                    refresh['user_email'] = user.email
                    refresh['username'] = user.username

                    response = JsonResponse({
                        'status': 'success',
                        'message': 'Login successful',
                        'username': user.username,
                        'email': user.email,
                        'access': str(refresh.access_token),
                        'refresh': str(refresh),
                    })
                    
                    # Add CORS headers
                    response["Access-Control-Allow-Origin"] = "*"
                    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
                    response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                    
                    return response
                else:
                    return JsonResponse({'error': 'Invalid credentials'}, status=400)
            return JsonResponse({'error': 'Invalid data'}, status=400)
        except Exception as e:
            print(f"Login error: {str(e)}")
            return JsonResponse({'error': 'Server error'}, status=500)

    def options(self, request):
        response = JsonResponse({}, status=200)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

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
        # Hash the password with bcrypt before saving
        password = request.data.get('password')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # Store user data temporarily in UserAuth model
        user_auth = UserAuth(
            username=request.data.get('username'),
            email=email,
            password=hashed_password.decode('utf-8')  # Save as a string in DB
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