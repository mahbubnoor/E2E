# Pre-requisite tools
1. Install Kaldi
2. Install ESPnet

#/E2E_ASR
1. start exceution of ASR recipe by typing run.sh in "/espnet/egs/basr/asr_practice_1/run.sh" until Dictionary and JSON format data preparation stages.
2. python gen_fbank.py to generate the FBank feature as the input for ASR model training.
3. type command "copy-feats ark,t:fbank.txt ark,scp:fbank.ark,fbank.scp"
4. python change_data.py to modify fbank feature in data.json.
5. run /espnet/egs/basr/asr_practice_1/run.sh again but from the training stage of ASR model with new fbank.
6. copy the pre-trained model from espnet by the following command:
cp ../espnet/egs/basr/asr_practice_1/exp/train_nodev_pytorch_train/results/model.loss.best.entire E2E_SE/data/newctcloss.model.acc.best.entire.pth

#/E2E_SE
1. modify data PATHS in the run.sh
2. sh run.sh 