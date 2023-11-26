import torchaudio
import torch
from audiolm_pytorch import SoundStream

model = SoundStream(
    codebook_size=1024,
    rq_num_quantizers=8
)

state_dict = torch.load('results/soundstream/soundstream_16khz-20230425.ckpt')['state_dict']

model.load_state_dict(state_dict)

x, sr = torchaudio.load('test.flac')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
with torch.no_grad():
    # y = model.encode(x)
    # # y = y[:, :, :4]  # if you want to reduce code size.
    # z = model.decode(y)
    recons = model(x, return_recons_only = True)

    for ind, recon in enumerate(recons.unbind(dim = 0)):
        filename = str('test_output.flac')
        torchaudio.save(filename, recon.cpu().detach(), sr)

# torchaudio.save('test_output.flac', z, sr)