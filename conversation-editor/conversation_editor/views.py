from django.shortcuts import render
from conversation_editor.models import Domain, Chat, Label
from django.conf import settings
# Create your views here.


def index(request):

    return render(request, 'home.html', {
        'page': 'include/main.html'
    })


def domain(request, pk):

    try:
        domain = Domain.objects.get(pk=pk)
    except Domain.DoesNotExist:
        raise Exception("Item doesn't exists")
    title = domain.domainName
    return render(request, 'home.html', {'page': 'include/domain.html',
                                            'title': title})


def conversation_editor(request, type):
    media = settings.MEDIA_ROOT
    with open(media+"\\health\\texts\\Text1.txt") as file:
        text = file.read()
    with open(media+"\\health\\labels\\Text1.txt") as file:
        labels = file.read()
    action_conversation = "New"
    if type == "edit":
        action_conversation = "Edit"
    return render(request, 'home.html', {'page': 'include/conversation_editor.html',
                                            'action_conversation': action_conversation,
                                            'text': text,
                                            'labels': labels})
