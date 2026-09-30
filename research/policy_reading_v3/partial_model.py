"""Resource amendment: train the same upper two encoder layers in all conditions."""
from model import Reader as FullReader

class Reader(FullReader):
    def __init__(self, checkpoint, mode):
        super().__init__(checkpoint, mode)
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = name.startswith(('encoder.layer.10.', 'encoder.layer.11.'))
