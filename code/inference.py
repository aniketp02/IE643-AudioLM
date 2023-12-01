from audiolm_pytorch import EncodecWrapper, HubertWithKmeans, SemanticTransformer, HubertWithKmeans, CoarseTransformer, FineTransformer, AudioLM

import os
import torch
import torchaudio
import urllib.request

import gradio as gr

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin'
coarse_transformer_ckpt = 'results/coarse/coarse.transformer.108000.pt'
fine_transformer_ckpt = 'results/fine/fine.transformer.5600.pt'
semantic_transformer_ckpt = 'results/semantic/semantic.transformer.500.pt'


if not os.path.isdir("hubert"):
  os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
  urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
  urllib.request.urlretrieve(hubert_quantizer_download, f"./{hubert_quantizer}")


wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
).cuda()
print("Load wav2vec")

encodec = EncodecWrapper(
    target_sample_hz=24000,
    num_quantizers=8
).cuda()
print("Load encodec")

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
).cuda()
semantic_transformer.load(f"./{semantic_transformer_ckpt}")
print("Load semantic transformer")


coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
).cuda()
coarse_transformer.load(f"./{coarse_transformer_ckpt}")
print("Load coarse tranformer")

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
).cuda()
fine_transformer.load(f"./{fine_transformer_ckpt}")
print("Load Fine transformer")

# Everything together
audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = encodec,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer,
)

sample_rate = 24000

# generated_wav = audiolm(batch_size = 1)                         # unconditional generation of audio
# output_path = 'out_unconditional.wav'
# torchaudio.save(output_path, generated_wav.cpu(), sample_rate)

# generated_wav_with_prime = audiolm(prime_wave_path = '../data/inference/continuation_1_prompt.wav')        # generating continuation with given audio prompt [note: if `prime_wave` given instead of path then `prime_wave_input_sample_hz` should also be given]
# output_path = 'out_continuation_1.wav'
# torchaudio.save(output_path, generated_wav_with_prime.cpu(), sample_rate)

# generated_wav_with_prime_piano = audiolm(prime_wave_path = '../data/inference/piano_continuation_1_prompt.wav')        # generating continuation with given audio prompt [note: if `prime_wave` given instead of path then `prime_wave_input_sample_hz` should also be given]
# output_path = 'out_piano_continuation_1.wav'
# torchaudio.save(output_path, generated_wav_with_prime_piano.cpu(), sample_rate)

# generate audio (not continuations) from ground truth semantic tokens extracted from input audio
# generated_wav_acoustic_generation = audiolm(                    
#     prime_wave_path = '../data/inference/acoustic_gen_1_orig.wav',
#     acoustic_generation = True
#     )
# output_path = 'out_acoustic_gen_1.wav'
# torchaudio.save(output_path, generated_wav_acoustic_generation.cpu(), sample_rate)

# generated_wav_with_text_condition = audiolm(                    # generate audio from input text
#     text = ['chirping of birds and the distant echos of bells'],
#     # return_coarse_generated_wave = True,
#     )
# output_path = "out_text2audio_1.wav"
# torchaudio.save(output_path, generated_wav_with_text_condition.cpu(), sample_rate)


def generate_audio(input_text, input_audio):
    if input_text:
        print("Input text is : ", input_text)
        generated_wav = audiolm(
            text=[input_text]
        )
    elif input_audio:
        prime_wave_input_sample_hz, prime_wave = input_audio
        sig_wav = torch.FloatTensor(prime_wave)
        sig_wav = torch.reshape(sig_wav, (1, -1))
        print(sig_wav.shape)

        generated_wav = audiolm(
            prime_wave_input_sample_hz=prime_wave_input_sample_hz,
            prime_wave=sig_wav
        )
    else:
        generated_wav = audiolm(batch_size = 1)  

    return (sample_rate, generated_wav.cpu().numpy().flatten())

iface = gr.Interface(
    title="Diffuser-AudioLM",
    fn=generate_audio,
    inputs=[gr.Textbox(), gr.Audio()],
    outputs=gr.Audio(),
    # live=True,
)

iface.launch()
