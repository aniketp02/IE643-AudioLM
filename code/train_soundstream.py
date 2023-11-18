import torchaudio
from audiolm_pytorch import SoundStream, SoundStreamTrainer

torchaudio.set_audio_backend('sox')

dataset_folder = '/home/ubuntu/IE643/project/data/wav_dir'

soundstream = SoundStream(
		codebook_size=1024,
		rq_num_quantizers=8
		)

soundstream_trainer = SoundStreamTrainer(
			soundstream,
			folder=dataset_folder,
			batch_size=4,
			grad_accum_every=8,
			data_max_length=320*32,
			save_results_every=2,
			save_model_every=4,
			num_train_steps=9
			).cuda()

soundstream_trainer.train()
