from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images/')
    # uploaded_at = models.DateTimeField()
    # id = models.UUIDField(primary_key = True, default=uuid.uuid4, unique = True)

    def __str__(self):
        return self.name

    def get_image(self):
        return self.image
