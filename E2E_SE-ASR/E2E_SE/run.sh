stage=2
stop_stage=3

# Input waveform data
TRAIN_NOISY="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_2/DATA/bopenslr_se/train/n_tr/"
TRAIN_CLEAN="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_2/DATA/bopenslr_se/train/cl_tr/"
TEST_NOISY="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_2/DATA/bopenslr_se/test/n/"
TEST_CLEAN="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_2/DATA/bopenslr_se/test/cl/"

# Output spectrum path
OUT_PATH="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_2/E2E_SE_output"
mkdir $OUT_PATH

#if [ ${stage} -eq 0 ]; then 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0 : Data preprocessing"
    python preprocess/gen_npy.py  --data 'trdata' --noisy_wav_path $TRAIN_NOISY --clean_wav_path $TRAIN_CLEAN --out_path $OUT_PATH
    python preprocess/gen_npy.py  --data 'tsdata' --noisy_wav_path $TEST_NOISY --clean_wav_path $TEST_CLEAN --out_path $OUT_PATH
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1 : Training"
    python secode/main.py --mode 'train' --out_path $OUT_PATH --train_clean $TRAIN_CLEAN #--train_num 10
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2 : Testing"
    python secode/main.py --mode 'test' --out_path $OUT_PATH --test_noisy $TEST_NOISY --test_clean $TEST_CLEAN --after_alpha_epoch
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3 : Scoring"
    python preprocess/cal_score.py $OUT_PATH/Result/transformerencoder_03_MAP_epochs150_adam_l1_alpha0.0_after_alpha_epoch70_batch2_lr5e-05.csv $OUT_PATH/Score/
    #python preprocess/cal_score.py $OUT_PATH/Result/transformerencoder_03_MAP_epochs150_adam_l1_alpha9e-06_after_alpha_epoch70_batch2_lr5e-05.csv $OUT_PATH/Score/
    #python preprocess/cal_score_cer.py $OUT_PATH/Result/transformerencoder_03_MAP_epochs150_adam_l1_alpha0.001_after_alpha_epoch70_batch4_lr5e-05.csv $OUT_PATH/Score/
fi