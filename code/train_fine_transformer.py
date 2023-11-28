from audiolm_pytorch import FineTransformer, FineTransformerTrainer, EncodecWrapper

dataset_folder = '/kaggle/input/librispeech-train-clean-100/LibriSpeech/train-clean-100'
results_folder = 'results/fine'

encodec = EncodecWrapper(
    target_sample_hz=24000,
    num_quantizers=8
)

fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
    codebook_size=1024,
    dim=512,
    depth=6
)

fine_transformer_trainer = FineTransformerTrainer(
    transformer=fine_transformer,
    codec=encodec,
    folder=dataset_folder,
    results_folder=results_folder,
    grad_accum_every = 8,
    batch_size=8,
    data_max_length_seconds=7,
    save_results_every=1000,
    save_model_every=1000,
    num_train_steps=120000,
).cuda()

restart_train = False
model_path = ''

if restart_train:
    fine_transformer_trainer.load(model_path)

fine_transformer_trainer.train()