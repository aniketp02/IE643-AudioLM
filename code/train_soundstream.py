import torchaudio
from audiolm_pytorch import SoundStream, SoundStreamTrainer

torchaudio.set_audio_backend('sox')

dataset_folder = '/home/ubuntu/IE643/project/data/small'
results_folder = 'results/soundstream'

soundstream = SoundStream(
		codebook_size=1024,
		rq_num_quantizers=8
		)

soundstream_trainer = SoundStreamTrainer(
			soundstream,
			folder=dataset_folder,
			results_folder=results_folder,
			batch_size=2,
			grad_accum_every=8,
			data_max_length_seconds=7,
			save_results_every=100,
			save_model_every=100,
			num_train_steps=20000
			).cuda()

soundstream_trainer.train()
