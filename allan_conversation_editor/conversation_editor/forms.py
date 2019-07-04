from django import forms
from conversation_editor.models import Domain, Chat, Label, ChatLine


class ChatForm(forms.Form):
    title = forms.CharField(label='Conversation title', max_length=100)


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

