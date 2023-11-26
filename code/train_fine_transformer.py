from audiolm_pytorch import SoundStream, FineTransformer, FineTransformerTrainer

soundstream_ckpt = 'results/soundstream.8.pt'
dataset_folder = '/home/ubuntu/IE643/project/data/wav_dir'
results_folder = 'results/fine'

soundstream = SoundStream(
    codebook_size=1024,
    rq_num_quantizers=8
)

soundstream.load(f'{soundstream_ckpt}')

fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
    codebook_size=1024,
    dim=512,
    depth=6
)

fine_transformer_trainer = FineTransformerTrainer(
    transformer=fine_transformer,
    codec=SoundStream,
    folder=dataset_folder,
    results_folder=results_folder,
    batch_size=1,
    data_max_length_seconds=7,
    num_train_steps=9
).cuda()

fine_transformer_trainer.train()