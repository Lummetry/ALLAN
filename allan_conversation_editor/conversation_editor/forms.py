from django import forms
from conversation_editor.models import Domain, Chat, Label, ChatLine


class ChatForm(forms.ModelForm):

    class Meta:
        model = Chat
        fields = ('title', 'txt_upload', 'lbl_upload', 'domain', 'created_user')
        widgets = {
            'title': forms.TextInput(
                attrs={'class': 'form-control',
                       'id': 'id_title',
                       'title': 'title',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
            'txt_upload': forms.FileInput(
                attrs={'class': 'form-control',
                       'id': 'id_txt_upload',
                       'title': 'txt_upload',
                       'required': False,
                       'placeholder': 'Say something...'}
            ),
            'lbl_upload': forms.FileInput(
                attrs={'class': 'form-control',
                       'id': 'id_lbl_upload',
                       'title': 'lbl_upload',
                       'required': False,
                       'placeholder': 'Say something...'}
            ),
            'domain': forms.TextInput(
                attrs={'class': 'input-hidden',
                       'id': 'id_domain',
                       'title': 'domain',
                       'required': True,
                       'placeholder': 'Say something...'}
            ),
            'created_user': forms.TextInput(
                attrs={'class': 'input-hidden',
                       'id': 'id_created_user',
                       'title': 'created_user',
                       'required': True,
                       'placeholder': 'Say something...'}
            )
        }

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

