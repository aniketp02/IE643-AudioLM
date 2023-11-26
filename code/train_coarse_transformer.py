from audiolm_pytorch import HubertWithKmeans, SoundStream, EncodecWrapper, CoarseTransformer, CoarseTransformerTrainer

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
# soundstream_ckpt = 'results/soundstream.8.pt'
dataset_folder = '/home/ubuntu/IE643/project/data/small'
results_folder = 'results/coarse'

wav2vec = HubertWithKmeans(
    checkpoint_path=f'{hubert_ckpt}',
    kmeans_path=f'{hubert_quantizer}'
)

# soundstream = SoundStream(
#     codebook_size=1024,
#     rq_num_quantizers=8
# )

# soundstream.load(f'{soundstream_ckpt}')

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
    batch_size=4,
    data_max_length_seconds=7,
    save_results_every=100,
    save_model_every=100,
    num_train_steps=20000,
).cuda()

coarse_transformer_trainer.train()