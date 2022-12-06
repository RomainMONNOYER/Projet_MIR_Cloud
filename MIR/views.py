from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def index(request):
    latest_question_list = "This is my question"
    template = loader.get_template('test.html')
    context = {
        'latest_question_list': [latest_question_list],
    }
    return HttpResponse(template.render(context, request))