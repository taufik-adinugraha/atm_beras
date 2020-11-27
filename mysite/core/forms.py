from django import forms
from .models import DataDiri



class Formulir(forms.ModelForm):
    class Meta:
        model = DataDiri
        fields = ('nama', 'alamat', 'foto')

    def clean_nama(self):
    	nama = self.cleaned_data.get('nama')
    	tmp1 = nama.lower().split(' ')
    	tmp1 = ''.join(tmp1)
    	for instance in DataDiri.objects.all():
    		tmp0 = instance.nama.lower().split(' ')
    		tmp0 = ''.join(tmp0)
    		if tmp0 == tmp1:
    			raise forms.ValidationError(f'Nama {nama.lower()} Sudah Terdaftar')
    	return nama
