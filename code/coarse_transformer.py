from audiolm_pytorch import HubertWithKMeans, SoundStream, CoarseTransformer, CoarseTransformerTrainer

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
soundstream_ckpt = 'results/soundstream.8.pt'
dataset_folder = '/home/ubuntu/IE643/project/data/wav_dir'

wav2vec = HubertWithKMeans(
    checkpoint_path=f'{hubert_ckpt}',
    kmeans_path=f'{hubert_quantizer}'
)

soundstream = SoundStream(
    codebook_size=1024,
    rq_num_quantizers=8
)

soundstream.load(f'{soundstream_ckpt}')

coarse_transformer = CoarseTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    codebook_size=1024,
    num_coarse_quantizers=3,
    dim=512,
    depth=6
)

coarse_transformer_trainer = CoarseTransformerTrainer(
    transformer=CoarseTransformer,
    codec=soundstream,
    wav2vec=wav2vec,
    folder=dataset_folder,
    batch_size=1,
    data_max_length=320*32,
    save_results_every=2,
    save_model_every=4,
    num_train_steps=9
).cuda()

coarse_transformer_trainer.train()