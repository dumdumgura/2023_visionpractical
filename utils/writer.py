import os
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, result_path):
        self.result_path = result_path

        self.writer_trn = SummaryWriter(os.path.join(result_path, "train"))
        self.writer_val = SummaryWriter(os.path.join(result_path, "valid"))
        self.writer_val_ema = SummaryWriter(os.path.join(result_path, "valid_ema"))

    def _get_writer(self, mode):
        if mode == "train":
            writer = self.writer_trn
        elif mode == "valid":
            writer = self.writer_val
        elif mode == "valid_ema":
            writer = self.writer_val_ema
        else:
            raise ValueError(f"{mode} is not valid..")

        return writer

    def add_mesh(self,mode,tag,vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        writer = self._get_writer(mode)
        writer.add_mesh(tag,vertices, colors, faces, config_dict, global_step, walltime)

    def add_scalar(self, tag, scalar, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_scalar(tag, scalar, epoch)

    def add_image(self, tag, image, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_image(tag, image, epoch)

    def add_text(self, tag, text, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_text(tag, text, epoch)

    def add_audio(self, tag, audio, mode, sampling_rate=16000, epoch=0):
        writer = self._get_writer(mode)
        writer.add_audio(tag, audio, epoch, sampling_rate)

    def close(self):
        self.writer_trn.close()
        self.writer_val.close()
        self.writer_val_ema.close()
