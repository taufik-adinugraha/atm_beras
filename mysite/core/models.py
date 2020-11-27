from django.db import models
import os


def path_and_rename(instance,filename):
	upload_to = 'foto'
	ext = filename.split('.')[-1]
	filename = instance.nama.lower().split(' ')
	filename = ''.join(filename)
	filename = '{}.{}'.format(filename, ext)
	return os.path.join(upload_to, filename)



class DataDiri(models.Model):
    nama = models.CharField(max_length=100)
    alamat = models.CharField(max_length=100)
    foto = models.ImageField(upload_to=path_and_rename)
    print(foto)

    def __str__(self):
    	return self.nama

    def delete(self, *args, **kwargs):
        self.foto.delete()
        super().delete(*args, **kwargs)




