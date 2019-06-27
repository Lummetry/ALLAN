from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('', views.index, name='home'),
    path('domain/<int:pk>', views.domain, name='domain'),
    path('conversation_editor/<str:type>', views.conversation_editor, name='conversation_editor')
]