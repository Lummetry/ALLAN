from django import forms
from conversation_editor.models import Domain, Chat, Label, ChatLine


class ChatForm(forms.ModelForm):

    class Meta:
        model = Chat
        fields = ('title', 'txt_upload', 'lbl_upload', 'domain', 'created_user')


class ChatLineForm(forms.ModelForm):
    class Meta:
        model = ChatLine
        # exclude = ['author', 'updated', 'created', ]
        fields = ['message', 'chat', 'label', 'parent']
        widgets = {
            'message': forms.TextInput(
                attrs={'class': 'form-control',
                       'id': 'id_message',
                       'title': 'message',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
            'chat': forms.TextInput(
                attrs={'class': 'input-hidden',
                       'id': 'id_chat',
                       'title': 'Your name',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
            'label': forms.TextInput(
                attrs={'class': 'input-hidden',
                       'id': 'id_label',
                       'title': 'Your name',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
            'parent': forms.TextInput(
                attrs={'class': 'input-hidden',
                       'id': 'id_parent',
                       'title': 'Your name',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
        }

