from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import Formulir
from .models import DataDiri
# from .filters import DataFilter


class Home(TemplateView):
    template_name = 'home.html'


def list(request):
    # daftar = DataDiri.objects.all()
    daftar = DataDiri.objects.order_by('-id')
    return render(request, 'list.html', {
        'daftar': daftar
    })


def register(request):
    if request.method == 'POST':
        form = Formulir(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('list')
    else:
        form = Formulir()
    return render(request, 'upload.html', {
        'form': form
    })


def delete(request, pk):
    if request.method == 'POST':
        dat = DataDiri.objects.get(pk=pk)
        dat.delete()
    return redirect('list')

