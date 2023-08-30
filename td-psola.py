import librosa
import numpy as np
import matplotlib.pyplot as plt
import samplerate
import soundfile as sf
from math import pow
from pdb import set_trace

def resample(seq, rate):
    for i in np.linspace(0, len(seq), rate):
        return seq[int(i)]

def psola(
        y: np.ndarray,
        rate: float,
        pitch_shift_factor: float = 1,
        formant_shift_factor: float = 1
):

    frame_length = 512
    
    f0, is_voiced, voiced_p = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=rate,
        frame_length=frame_length,
        fill_na=np.nan)

    print("pyin done")

    times = librosa.times_like(f0, hop_length=frame_length//4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

    x_offset = 0
    next_peak = 0
    curr_peak = 0

    output = np.zeros(len(y))
    acc = []
    for frame_index, x in enumerate(range(0, len(y), frame_length//4)):
        if is_voiced[frame_index]:
            period = int(sr/f0[frame_index])
            pitch_shift_factor = 440/f0[frame_index]

                
            # for i in range(x_offset+x, frame_length//4+x, period):
            #     try:
            #         temp = samplerate.resample(y[i:i+2*period], 1/formant_shift_factor)
            #     except:
            #         temp = samplerate.resample(y[i:i+period], 1/formant_shift_factor)
            #     acc.append(temp*np.hamming(len(temp)))
            #     # if i+len(temp) > len(output):
            #     #     pass
            #     # else:
            #     #     output[
            #     #         i:i+len(temp)
            #     #     ] += temp*np.hamming(len(temp))
            #     x_offset += period
            should_exit = False
            while not should_exit:
                if curr_peak+2*period < x+frame_length//4:
                    next_peak = np.argmax(np.abs(y[curr_peak+period-20:curr_peak+2*period]))
                    next_peak += curr_peak+period-20
                else:
                    next_peak = len(y) - 1
                    should_exit = True

                temp = samplerate.resample(y[curr_peak:next_peak], 1/formant_shift_factor)
                acc.append(temp*np.hamming(len(temp)))

                curr_peak = next_peak

            # for output_index, input_index in zip(
            #         np.linspace(
            #             x_offset+x,
            #             frame_length//4+x,
            #             num=(cnt:=int(len(acc)*pitch_shift_factor))
            #         ),
            #         np.linspace(
            #                 0,
            #                 len(acc),
            #                 num=cnt
            #             )
            # ):
            #     try:
            #         a = acc[int(input_index)]
            #         start = int(output_index)
            #         output[start:start+len(a)] += a
            #     except:
            #         pass

            x_offset -= frame_length//4
        else:
            if acc:
                print(len(acc))
                for output_index, input_index in zip(
                        np.linspace(
                            x_offset+x,
                            frame_length//4+x,
                            num=(cnt:=int(len(acc)*pitch_shift_factor))
                        ),
                        np.linspace(
                            0,
                            len(acc),
                            num=cnt
                        )
                ):
                    try:
                        a = acc[int(input_index)]
                        start = int(output_index)
                        output[start:start+len(a)] += a
                    except:
                        pass
            
            acc = []
            x_offset = 0
            output[x:x+frame_length//4] += y[x:x+frame_length//4]
            next_peak = x+frame_length//4
            curr_peak = x+frame_length//4
                
    return output

if __name__ == "__main__":
    y, sr = librosa.load("harvard.wav")
    print(y.shape)

    out = psola(y[:], sr, 4)
    sf.write("out.wav", out, sr)

    f0, is_voiced, voiced_p = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        fill_na=np.nan)

    times = librosa.times_like(f0)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation: After')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()

