from django import forms
from conversation_editor.models import Domain, Chat, Label


class ChatForm(forms.Form):
    title = forms.CharField(label='Conversation title', max_length=100)
    location = forms.CharField(label='Path', max_length=255)
    text = forms.Text(label='Converstaion')
