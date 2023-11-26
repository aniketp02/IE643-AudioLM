from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer
import urllib
import os

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
dataset_folder = '/home/ubuntu/IE643/project/data/small'
results_folder = 'results/semantic'

if not os.path.isdir('hubert'):
    os.makedirs('hubert')

if not os.path.isfile(f'{hubert_ckpt}'):
    hubert_ckpt_download = f'https://dl.fbaipublicfiles.com/{hubert_ckpt}'
    urllib.request.urlretrieve(hubert_ckpt_download, f'{hubert_ckpt}')

if not os.path.isfile(f'{hubert_quantizer}'):
    hubert_quantizer_download = f'https://dl.fbaipublicfiles.com/{hubert_quantizer}'
    urllib.request.urlretrieve(hubert_quantizer_download, f'{hubert_quantizer}')

wav2vec = HubertWithKmeans(
    checkpoint_path=f'{hubert_ckpt}',
    kmeans_path=f'{hubert_quantizer}'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    dim=1024,
    depth=6,
)

semantic_transformer_trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    folder=dataset_folder,
    results_folder=results_folder,
    batch_size=4,
    data_max_length_seconds=7,
    save_results_every=100,
	save_model_every=100,
    num_train_steps=20000
).cuda()

semantic_transformer_trainer.train()