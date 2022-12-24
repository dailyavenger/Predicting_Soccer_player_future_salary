import numpy as np
import pandas as pd

data_num = 0

data = np.empty((0, 18), float)

for i in range(1, 522):
    url = "./datasets/players/" + str(i) + ".csv"

    t = pd.read_csv(url, index_col = 0).astype(float)

    t_numpy = t.to_numpy()[:, :]
    t_numpy = t_numpy.T

    data = np.append(data, t_numpy, axis = 0)


print(data.shape)

mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)

pay_mean = mean[1]
pay_std = std[1]

# 맨 뒤에 연봉 데이터가 들어가게끔 수정
mean = np.delete(mean, 1) # 연봉 데이터 값만 삭제
std = np.delete(std, 1) # 연봉 데이터 값만 삭제

mean = np.append(mean, pay_mean)
std = np.append(std, pay_std)

mean_std = np.concatenate([mean, std], axis = 0)
mean_std = mean_std.reshape(2, 18)

np.save("./datasets/mean_std", mean_std)

# 순서
#1 UEFA
#2. Games played
#3. Goals
#4. Assists
#5. Efficiency
#6. Own goals
#7. Goal against
#8. Clean sheets
#9. Minutes played
#10. Match start
#11. Substitute
#12. On the game sheet
#13. Yellow card
#14. Red card
#15. Win
#16. Draw
#17. Lose
#18. 연봉
