import numpy as np
import pandas as pd

window_size = 7 # 몇 년을 window 사이즈로 잡을 것인지

encoder_x = np.empty((0, 17), float)
decoder_x = np.empty((0, window_size), float)
y = np.empty((0, window_size), float)

for i in range(1, 522):
    url = "./datasets/players/" + str(i) + ".csv"

    t = pd.read_csv(url, index_col = 0).astype(float)
    print(t)

    t_numpy = t.to_numpy()[:, :]
    t_numpy = t_numpy.T

    # 만들 수 있는 데이터가 없는 경우
    if(t_numpy.shape[0] < window_size) :
        continue

    for i in range(window_size - 1, t_numpy.shape[0] - 1):
        tmp = t_numpy[i - (window_size - 1) : i + 1] # 3개년치 데이터를 가지고 있음
        tmp = np.delete(tmp, 1, axis = 1) # 연봉 데이터 값만 삭제
        encoder_x = np.append(encoder_x, tmp, axis = 0)
        decoder_x = np.append(decoder_x, t_numpy[i - (window_size - 1) : i + 1, 1])
        y = np.append(y, t_numpy[i - (window_size - 1) + 1 : i + 2, 1])


encoder_x = np.reshape(encoder_x, (-1, window_size * 17))
decoder_x = np.reshape(decoder_x, (-1, window_size))

x = np.concatenate((encoder_x, decoder_x), axis=1)
y = np.reshape(y, (-1, window_size))

print(x.shape)
print(y.shape)

np.save("./datasets/saved_x_" + str(window_size), x)
np.save("./datasets/saved_y_"+ str(window_size), y)