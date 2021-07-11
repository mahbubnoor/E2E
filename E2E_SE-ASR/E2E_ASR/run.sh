ASR_PATH="/Data/user_mhb/ESPnet/espnet/egs/basr/asr_practice_1"
CLEAN_BOpenSlr="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_1/DATA/bopenslr/"
DATA_PATH="/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_1/data/BOpenSlr_fbank"
PWD_PATH='/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_1/E2E_ASR'
SE_PATH='/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_1/E2E_SE'

step=4
stop_step=4
if [ ${step} -le 0 ] && [ ${stop_step} -ge 0 ]; then
    echo "step 0 : Execute ASR recipe until stage 2 to get the features"
    cd $ASR_PATH
    ./run.sh --stop-stage 2 
fi

cd $PWD_PATH/
if [ ${step} -le 1 ] && [ ${stop_step} -ge 1 ]; then
    echo "step 1 : generate normalized fbank"
    mkdir out/
    python make_list.py  $CLEAN_BOpenSlr out/BOpenSlr_filelist.txt
    python gen_fbank.py  out/BOpenSlr_filelist.txt $DATA_PATH/bopenslr_fbank.txt
    copy-feats ark,t:$DATA_PATH/bopenslr_fbank.txt ark,scp:$DATA_PATH/bopenslr_fbank.ark,$DATA_PATH/bopenslr_fbank.scp
fi

if [ ${step} -le 2 ] && [ ${stop_step} -ge 2 ]; then
    echo "step 2 : change data.json for matching both ASR and SE"
    python change_data.py $DATA_PATH/bopenslr_fbank.scp $ASR_PATH/dump/test/deltafalse/data.json out/data_test.json
    python change_data.py $DATA_PATH/bopenslr_fbank.scp $ASR_PATH/dump/train_dev/deltafalse/data.json out/data_train_dev.json
    python change_data.py $DATA_PATH/bopenslr_fbank.scp $ASR_PATH/dump/train_nodev/deltafalse/data.json out/data_train_nodev.json
    cp out/data_test.json $ASR_PATH/dump/test/deltafalse/data.json
    cp out/data_train_dev.json $ASR_PATH/dump/train_dev/deltafalse/data.json
    cp out/data_train_nodev.json $ASR_PATH/dump/train_nodev/deltafalse/data.json
    cp out/data_test.json $SE_PATH/data/
    cp out/data_train_dev.json $SE_PATH/data/
    cp out/data_train_nodev.json $SE_PATH/data/
fi
if [ ${step} -le 3 ] && [ ${stop_step} -ge 3 ]; then
    echo "step 3 : train & test ASR model"
    cd $ASR_PATH/
    ./run.sh --stage 3 
fi
if [ ${step} -le 4 ] && [ ${stop_step} -ge 4 ]; then
    echo "step 4 : copy ASR model from espnet and run E2E-SE"
    cd $SE_PATH
    cp $ASR_PATH/exp/train_nodev_pytorch_train_mtlalpha0.3/results/model.acc.best.entire data/newctcloss.model.acc.best.entire.pth
    sh run.sh
fi