from matplotlib import colors
import librosa
import os
import glob
import torch
import torchaudio
import numpy as np

def wav_to_mel(wav, window, stft_args, mel_filterbank):
    spec = torch.stft(wav, window=window, normalized=True, 
                    return_complex=True, **stft_args)
    magnitude = torch.abs(spec)
    mel_spec = mel_filterbank(magnitude)
    mel_spec = mel_spec / torch.max(mel_spec)
    return mel_spec

def sound_render( s, sr=16000, show='stft', name=''):
    import base64
    from io import BytesIO
    from matplotlib.pyplot import figure, gca, gcf, close

    # Get the visual if requested
    if show is not None:
        figure( figsize=(2.5,.75))
        if show == 'stft':
            sz = sr//64
            gca().specgram( s, Fs=sr, NFFT=sz, noverlap=sz-sz//4, scale='linear', norm=colors.PowerNorm( gamma=0.2))
        elif show == 'wave':
            gca().plot( s)
        elif show == 'spec':
            from numpy import log1p
            from numpy.fft import rfft
            gca().plot( abs( rfft( s))**.5)
        elif show == 'mel':
            mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sr, n_stft=256)
            window = torch.hann_window(510, periodic=True)
            stft_args = dict(n_fft=510, hop_length=256, center=True)
            S = wav_to_mel(wav=torch.tensor(s),
                                  window=window,
                                  stft_args=stft_args,
                                  mel_filterbank=mel_filterbank)
            S = S.squeeze().numpy()[:, :256]
            S_dB = librosa.power_to_db(S, ref=np.max)
            # invesre the S_dB y axis
            S_dB = S_dB[::-1, :]
            gca().imshow(S_dB) 
            
        gca().axis( 'tight')
        gca().axis( False)
        gcf().tight_layout( pad=0)
        imf = BytesIO()
        gcf().savefig( imf, format='png')
        m = base64.b64encode( imf.getvalue()).decode('utf-8') + '\n'
        d = [f'<img src="data:image/jpeg;base64,{m}"  style="background-color:white;">']
        close()

    # Make a table with name / optional visual / sound player
    import tabulate
    from IPython.display import Audio
    t = [[name]]
    if show is not None:
        t += [d]
    t += [[Audio( s/abs( s).max(), rate=sr)._repr_html_()[3:].replace( 'controls', 'controls style="width: 250px; height: 24px;"')]]

    return tabulate.tabulate( t, tablefmt='unsafehtml')

def soundgrid( *k, sr=16000, r_header=[], c_header=[], show='stft'):
    import tabulate
    t = list( zip( *[[sound_render(s.squeeze(), sr, show=show)._repr_html_() for s in p] for p in k]))
    for i in range(len(t)):
        t[i] = list(t[i])
        t[i].insert(0, c_header[i])
        t[i] = tuple(t[i])
    return tabulate.tabulate( t, tablefmt='unsafehtml', headers=r_header, stralign='center')



root_dir = '/Users/cooper/Downloads/PromptSep/audios/text_cond'
sample_list = [['real_0_mix', 'real_0_mix', 'real_0_mix', 'real_1_mix', 'real_1_mix', 'minions', 'nyc_walk_1', 'nyc_walk_1', 'beijing vlog', 'beijing vlog'], 
               ['fireworks', 'woman shouting', 'hissing sound', 'finger snapping', 'male voice', 'glass breaking and explosion', 'speech_1', 'street ambiance_1', 'background music', 'mouse click sound effect']]
row_content = []
sample_rate = 44100

for row in sample_list:
    audios = []
    for col in row:
        filename = os.path.join(root_dir, f'{col}.wav')
        filename = glob.glob(filename)[0]
        x, sr = librosa.load(filename, sr=sample_rate)
        # pad or trim to 441000 samples
        if len(x) < 441000:
            x = np.pad(x, (0, 441000 - len(x)))
        else:
            x = x[:441000]
            
        assert sr == sample_rate
        audios.append(x)
    row_content.append(audios)
        
row_header = ['Text Prompt', 'Mixture', 'Ours']
c_header = ['Fireworks', 'Woman shouting', 'Hissing sound', 
            'Finger snapping', 'I want only the male voice, no finger snapping',
            'Glass breaking and explosion', 'Speech', 'Street ambiance',
            'Background music', 'Mouse click sound effect']
# Make three sound sets
# x = [sin( i*linspace( 0, 2*pi, 8000)**1.4) for i in [100, 200, 400, 800, 1600]]
# y = [sign( sin( i*linspace( 0, 2*pi, 8000)**1.4)) for i in [100, 200, 400, 800, 1600]]
# z = [n[i+1:] + n[:-i-1] for i,n in enumerate( random.randn( 5, 8000))]

# Pack them into a grid
sg = soundgrid(*row_content, sr=sample_rate, r_header=row_header, c_header=c_header, show='stft')

with open('hearme.html', 'w') as f:
    f.write(sg)