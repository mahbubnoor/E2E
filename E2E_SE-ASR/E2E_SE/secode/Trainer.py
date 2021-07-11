import torch.nn as nn
import torch
import pandas as pd
import os, sys
from tqdm import tqdm
import librosa, scipy
import pdb
import numpy as np
from scipy.io.wavfile import write as audiowrite
from utils.util import  check_folder, recons_spec_phase, cal_score, make_spectrum, get_cleanwav_dic, getfilename
maxv = np.iinfo(np.int16).max

def prepare_test(test_file, c_dict, device):
    name = test_file.split('/')[-1].split('-')[0] + '-' + test_file.split('/')[-1].split('-')[1]
    n_folder = '/'.join(test_file.split('/')[-2:-1])
    name = name.replace('.wav','')
    c_name = name.split('-')[0]+'-'+name.split('-')[1]+'.wav'
    #print('c_name: ',c_name, 'noisy: ',name,  'noisy folder: ',n_folder)

    c_folder=c_dict[name]
    clean_file= os.path.join(c_folder, c_name)
    #print('clean file: ',clean_file)
    #print('noisy file: ',test_file)

    n_wav,sr = librosa.load(test_file,sr=16000)
    c_wav,sr = librosa.load(clean_file,sr=16000)

    n_spec,n_phase,n_len = make_spectrum(y=n_wav)
    c_spec,c_phase,c_len = make_spectrum(y=c_wav)

    n_spec = torch.from_numpy(n_spec.transpose()).to(device).unsqueeze(0)
    c_spec = torch.from_numpy(c_spec.transpose()).to(device).unsqueeze(0)
    return n_spec, n_phase, n_len, c_wav, c_spec, c_phase, n_folder

    
def write_score(model, device, test_file, c_dict, enhance_path, ilen, y, score_path):
    n_spec, n_phase, n_len, c_wav, c_spec, c_phase, n_folder = prepare_test(test_file, c_dict,device)
    enhanced_spec = model.SEmodel(n_spec).cpu().detach().numpy()
    enhanced = recons_spec_phase(enhanced_spec.squeeze().transpose(),n_phase,n_len)
    # cal score
    s_pesq, s_stoi = cal_score(c_wav,enhanced)
    with open(score_path, 'a') as f:
        f.write(f'{test_file},{s_pesq},{s_stoi}\n')
    # write enhanced waveform
    out_path = f"{enhance_path}/{n_folder+'/'+test_file.split('/')[-1]}"
    check_folder(out_path)
    audiowrite(out_path,16000,(enhanced* maxv).astype(np.int16))

        
            
def test(model, device, noisy_path, clean_path, asr_dict, enhance_path, score_path, args):
    model = model.to(device)
    # load model
    model.eval()
    torch.no_grad()
    
    # load data
    if args.test_num is None:
        test_files = np.array(getfilename(noisy_path,"test"))
    else:
        test_files = np.array(getfilename(noisy_path,"test")[:args.test_num])

    c_dict = get_cleanwav_dic(clean_path)
    
    #open score file
   
    if os.path.exists(score_path):
        os.remove(score_path)
    
    check_folder(score_path)
    print('Save PESQ&STOI results to:', score_path)
    
    with open(score_path, 'a') as f:
        f.write('Filename,PESQ,STOI\n')

    print('Testing...')       
    for test_file in tqdm(test_files):
        name=test_file.split('/')[-1].replace('.wav','')
        ilen, y=asr_dict[name][0],asr_dict[name][1]
        write_score(model, device, test_file, c_dict, enhance_path, ilen, y, score_path)

    data = pd.read_csv(score_path)
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    with open(score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean)))+'\n')
