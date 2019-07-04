from django.shortcuts import render
from conversation_editor.models import Domain, Chat, Label, ChatLine
from django.contrib.auth.models import User
from django.conf import settings
from conversation_editor.forms import ChatForm, ChatLineForm
from django.shortcuts import redirect
import datetime
from django.http import HttpResponse
import json
from django.db.models import ProtectedError
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
import time

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
    chats = Chat.objects.all().filter(domain=domain)
    return render(request, 'home.html', {'page': 'include/domain.html',
                                         'title': title,
                                         'domain': pk,
                                         'chats': chats})


def add_edit_conversation(request, domain, id):
    domain_name = None
    chat_title = None
    fields = {}
    if id == 0:
        id = None
        fields['created_user'] = request.user
    try:
        domainD = Domain.objects.get(pk=domain)
        fields['domain'] = domainD
        domain_name = domainD.domainName
    except Domain.DoesNotExist:
        raise Exception("Item doesn't exists")
    if id is not None:
        try:
            chatEdit = Chat.objects.get(pk=id)
            chat_title = chatEdit.title
        except Domain.DoesNotExist:
            raise Exception("Item doesn't exists")
    action_conversation = "New"
    if id is not None:
        action_conversation = "Edit"
    if request.method == "POST":
        form = ChatForm(request.POST)
        if form.is_valid():
            fields['title'] = request.POST['title']
            fields['updated'] = datetime.datetime.now()
            chat, created = Chat.objects.update_or_create(
                id=id, defaults=fields
            )
            return redirect('domain', pk=domain)

    else:
        form = ChatForm(initial={'title': chat_title})
    return render(request, 'home.html', {'page': 'include/add_edit_conversation.html',
                                         'action_conversation': action_conversation,
                                         'domain_name': domain_name,
                                         'domain': domain,
                                         'form': form})


def conversation_editor(request, id):
    # media = settings.MEDIA_ROOT
    # with open(media+"\\health\\texts\\Text1.txt") as file:
    #    text = file.read()
    # with open(media+"\\health\\labels\\Text1.txt") as file:
    #    labels = file.read()
    try:
        chat = Chat.objects.get(pk=id)
        chat_title = chat.title
        domain_title = chat.domain
        domain_id = chat.domain.id
    except Chat.DoesNotExist:
        raise Exception("Item doesn't exists")
    try:
        chatLines = ChatLine.objects.filter(chat=chat).order_by('id')

    except ChatLine.DoesNotExist:
        raise Exception("Item doesn't exists")

    return render(request, 'home.html', {'page': 'include/conversation_editor.html',
                                         'chat_title': chat_title,
                                         'chat_id': id,
                                         'domain_title': domain_title,
                                         'domain_id': domain_id,
                                         'chatLines': chatLines,
                                         'form': ChatLineForm()})

def create_message(request):
    if request.method == 'POST':
        parent_id = request.POST.get('parent')
        chat_id = request.POST.get('chat')
        label_id = request.POST.get('label')
        message = request.POST.get('message')
        domain_id = request.POST.get('domain')
        label_name = request.POST.get('label_name')
        response_data = {}

        if int(label_id) == 0:
            try:
                domainD = Domain.objects.get(pk=domain_id)
            except Domain.DoesNotExist:
                raise Exception("Item doesn't exists")

            fields = {}
            fields['domain'] = domainD
            fields['name'] = label_name
            label1, created = Label.objects.update_or_create(
                    id=None, defaults=fields
                )
            label_id = label1.id
        try:
            chat = Chat.objects.get(pk=chat_id)
        except Chat.DoesNotExist:
            raise Exception("Item doesn't exists")

        try:
            label = Label.objects.get(pk=label_id)
        except Label.DoesNotExist:
            raise Exception("Item doesn't exists")

        try:
            parent = ChatLine.objects.get(pk=parent_id)
        except ChatLine.DoesNotExist:
            parent = None
        chatLine = ChatLine(chat=chat, created_user=request.user, parent=parent, message=message, label=label)
        chatLine.save()

        response_data['result'] = 'Create post successful!'

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )

def update_message(request, id):
    if request.method == 'POST':
        parent_id = request.POST.get('parent')
        chat_id = request.POST.get('chat')
        label_id = request.POST.get('label')
        message = request.POST.get('message')
        domain_id = request.POST.get('domain')
        label_name = request.POST.get('label_name')
        response_data = {}

        if int(label_id) == 0:
            try:
                domainD = Domain.objects.get(pk=domain_id)
            except Domain.DoesNotExist:
                raise Exception("Item doesn't exists")

            fields = {}
            fields['domain'] = domainD
            fields['name'] = label_name
            label1, created = Label.objects.update_or_create(
                    id=None, defaults=fields
                )
            label_id = label1.id
        try:
            chat = Chat.objects.get(pk=chat_id)
        except Chat.DoesNotExist:
            raise Exception("Item doesn't exists")

        try:
            label = Label.objects.get(pk=label_id)
        except Label.DoesNotExist:
            raise Exception("Item doesn't exists")

        try:
            parent = ChatLine.objects.get(pk=parent_id)
        except ChatLine.DoesNotExist:
            parent = None
        fields = {}
        fields['chat'] = chat
        fields['parent'] = parent
        fields['message'] = message
        fields['label'] = label
        chatLine, created = ChatLine.objects.update_or_create(
                    id=id, defaults=fields
                )
        chatLine.save()

        response_data['result'] = 'update post successful!'

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )

def get_message(request,id):
    response_data = {}
    try:
        chat_line = ChatLine.objects.get(pk=id)
    except ChatLine.DoesNotExist:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )

    response_data['chat_id'] = chat_line.chat_id
    response_data['parent_id'] = chat_line.parent_id if chat_line.parent_id else 0
    response_data['message'] = chat_line.message
    response_data['label_id'] = chat_line.label_id
    response_data['label_name'] = chat_line.label.name
    response_data['human'] = chat_line.human

    return HttpResponse(
        json.dumps(response_data),
        content_type="application/json"
    )


def delete_message(request, id):
    confirm = request.GET['confirm']
    print(confirm)
    id2 = None
    try:
        chatLine = ChatLine.objects.get(id=id)#.delete()
        if chatLine.human == True:
            txt = "<br>"+chatLine.message+"<br>"
            try:
                ch = ChatLine.objects.filter(id__gt=id, chat=chatLine.chat).order_by('id').first()
                print(ch)
                txt +=ch.message
                id2 = ch.id
            except ChatLine.DoesNotExist:
                parent = None
        else:
            txt = "<br>"
            try:
                ch = ChatLine.objects.filter(id__lt=id, chat=chatLine.chat).order_by('-id').first()
                print(ch)
                txt +=ch.message+"<br>"+chatLine.message
                id2 = ch.id
            except ChatLine.DoesNotExist:
                parent = None

        if confirm == 'false':
            return HttpResponse(
                json.dumps({"success": True, "confirmation": True, "message": txt}),
                content_type="application/json"
            )
    except ChatLine.DoesNotExist:
        error_message = "This object can't be deleted!!"
        return HttpResponse(
            json.dumps({"success": False, "error": error_message}),
            content_type="application/json"
        )
    if confirm == 'true':
        try:
            ChatLine.objects.filter(id__in=(id, id2)).delete()
            return HttpResponse(
                json.dumps({"success": True, "confirmation": True, "deleted": True}),
                content_type="application/json"
            )
        except ProtectedError:
            error_message = "This object can't be deleted!!"
            return HttpResponse(
                json.dumps({"success": False, "error": error_message}),
                content_type="application/json"
            )


def get_autocomplete_labels(request,id):
    response_data = []

    try:
        domain = Domain.objects.get(pk=id)
    except Domain.DoesNotExist:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )
    try:
        search = request.GET.get('s')
        labels = Label.objects.filter(domain=domain, name__icontains=search)
    except Label.DoesNotExist:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )
    for label in labels:
        data = {}
        data['id'] = label.id
        data['name'] = label.name
        data['s'] = request.GET.get('s')
        response_data.append(data)
    return HttpResponse(
        json.dumps(response_data),
        content_type="application/json"
    )



class ApiCreateConversationTypeView(APIView):

    def post(self,request):
        try:
            domain_id = request.POST['domain_id']
            fields = {}
            try:
                uid = User.objects.get(username="allan")
                fields['created_user'] = uid
            except User.DoesNotExist:
                raise Exception("User doesn't exists")

            try:
                domainD = Domain.objects.get(pk=domain_id)
                fields['domain'] = domainD
            except Domain.DoesNotExist:
                raise Exception("Domain doesn't exists")

            fields['title'] = 'chat_'+str(int(time.time()))
            fields['updated'] = datetime.datetime.now()
            chat, created = Chat.objects.update_or_create(
                id=None, defaults=fields
            )
            id_chat = chat.id
            return Response({"success": True, "message": None, "data": {"id": str(id_chat)}})
        except Exception as e:
            return Response({"success": False, "message": e.args[0], "data": None})


class ApiCreateMessageTypeView(APIView):

    def post(self,request):
        try:
            chat_id = request.POST['chat_id']
            human = request.POST['human']
            message = request.POST['message']
            label_name = request.POST['label']
            fields = {}
            domain = None
            try:
                uid = User.objects.get(username="allan")
                fields['created_user'] = uid
            except User.DoesNotExist:
                raise Exception("User doesn't exists")

            try:
                chat = Chat.objects.get(pk=chat_id)
                fields['chat'] = chat
                domain = chat.domain
            except Chat.DoesNotExist:
                raise Exception("Chat doesn't exists")

            try:
                label = Label.objects.get(
                    domain=domain,
                    name=label_name,
                )
            except Label.DoesNotExist:
                raise Exception("Label doesn't exists")
            chatLine = ChatLine(chat=chat,
                                created_user=fields['created_user'],
                                parent=None,
                                message=message,
                                label=label,
                                human=human)
            chatLine.save()

            return Response({"success": True, "message": None, "data": {"id": str(chatLine.id)}})
        except Exception as e:
            return Response({"success": False, "message": e.args[0], "data": None})



class ApiCreateLabelTypeView(APIView):

    def post(self,request):
        try:
            domain_id = 1
            label_name = request.POST['label']
            fields = {}
            domain = None
            try:
                uid = User.objects.get(username="allan")
                fields['created_user'] = uid
            except User.DoesNotExist:
                raise Exception("User doesn't exists")

            chat, created = Label.objects.get_or_create(name=label_name, domain_id=domain_id)


            return Response({"success": True, "message": None, "data": {"id": str(chat.id)}})
        except Exception as e:
            return Response({"success": False, "message": e.args[0], "data": None})
