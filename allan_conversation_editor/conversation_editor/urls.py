from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('', views.index, name='home'),
    path('domain/<int:pk>', views.domain, name='domain'),
    path('add_edit_conversation/<int:domain>/<int:id>', views.add_edit_conversation, name='add_edit_conversation'),
    path('conversation_editor/<int:id>', views.conversation_editor, name='conversation_editor'),
    path('create_message/', views.create_message, name='create_message'),
    path('update_message/<int:id>', views.update_message, name='update_message'),
    path('delete_message/<int:id>', views.delete_message, name='delete_message'),
    path('get_message/<int:id>', views.get_message, name='get_message'),
    path('get_autocomplete_labels/<int:id>', views.get_autocomplete_labels, name='get_autocomplete_labels'),
    path('api_create_conversation/', views.ApiCreateConversationTypeView.as_view(), name='api_create_conversation'),
    path('api_create_message/', views.ApiCreateMessageTypeView.as_view(), name='api_create_message'),
    path('api_create_label/', views.ApiCreateLabelTypeView.as_view(), name='api_create_label')
]