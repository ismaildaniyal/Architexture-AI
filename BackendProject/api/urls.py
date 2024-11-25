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
        path('signup/', views.SignupView.as_view(),),
        path('login/', views.LoginView.as_view(), name='login'),
        path('token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
        path('token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),

    ])

load_views()