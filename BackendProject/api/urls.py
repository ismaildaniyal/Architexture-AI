# from django.urls import path
# from api import views
# urlpatterens = [
#     path ('user/', views.UserList, name = 'user'),
# ]

# api/urls.py
from django.urls import path

from rest_framework_simplejwt import views as jwt_views
urlpatterns = []

def load_views():
    from . import views  # Local import
    urlpatterns.extend([
        path('user/', views.UserList.as_view(),),
        # path('signup/', views.SignupView.as_view(),),
        path('login/', views.LoginView.as_view(), name='login'),
        path('token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
        path('token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
        path('generate-otp/', views.GenerateOtpView.as_view(), name='generate-otp'),
        path('update-password/',views.StorePasswordView.as_view(), name='update-password'),
        path('verify-otp/', views.VerifyOtpView.as_view(), name='verify-otp'),
        path('signup/', views.SignupView1.as_view(), name='signup'),
        path('verify-email/<uidb64>/<token>/', views.VerifyEmailView.as_view(), name='verify_email'),
        path("process-houseplan/", views.HousePlanAPI.as_view(), name="process-houseplan"),
        path("retrive-data",views.UserChatAPIView.as_view()),
        path("delete-chat",views.DeleteUserChatAPIView.as_view()),
    ])
load_views()