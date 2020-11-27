import django_filters
from .models import DataDiri

class DataFilter(django_filters.FilterSet):

	class Meta:
		model = DataDiri
		fields = ('nama', 'alamat', 'foto')

