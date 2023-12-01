from audiolm_pytorch import HubertWithKmeans, SoundStream, EncodecWrapper, CoarseTransformer, CoarseTransformerTrainer

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
dataset_folder = '/kaggle/input/librispeech-train-clean-100/LibriSpeech/train-clean-100'
results_folder = 'results/coarse'

wav2vec = HubertWithKmeans(
    checkpoint_path=f'{hubert_ckpt}',
    kmeans_path=f'{hubert_quantizer}'
)

encodec = EncodecWrapper(
    target_sample_hz=24000,
    num_quantizers=8
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    codebook_size=1024,
    num_coarse_quantizers=3,
    dim=512,
    depth=6
)

coarse_transformer_trainer = CoarseTransformerTrainer(
    transformer=coarse_transformer,
    codec=encodec,
    wav2vec=wav2vec,
    folder=dataset_folder,
    results_folder=results_folder,
    grad_accum_every = 8,
    batch_size=8,
    data_max_length_seconds=7,
    save_results_every=100,
    save_model_every=1000,
    num_train_steps=100000,
).cuda()

restart_train = False
model_path = '/kaggle/input/coarse-transformer-pt-19900/coarse.transformer.19900.pt'

if restart_train:
    coarse_transformer_trainer.load(model_path)

coarse_transformer_trainer.train()
