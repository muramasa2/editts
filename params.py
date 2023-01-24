train_filelist_path = "resources/filelists/train.txt"
valid_filelist_path = "resources/filelists/valid.txt"
test_filelist_path = "resources/filelists/test.txt"
cmudict_path = "resources/cmu_dictionary"
add_blank = True
n_feats = 80
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
n_mels = 80
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
	@@ -28,11 +37,11 @@
pe_scale = 1  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = "logs/new_exp"
test_size = 4
n_epochs = 10000
batch_size = 16
learning_rate = 1e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2 * 22050 // 256)