import io

class NamedBytesIO(io.BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop("name", "buffer")
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name
