from tortoise.models import Model
from tortoise import fields


class User(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=100)
    balance = fields.FloatField(default=0.0)

    class Meta:
        table = "users"