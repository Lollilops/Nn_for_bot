# files = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt").read()
# files = files.replace("/", "")
# # files = files.replace(")", "")
# new_files = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt", "w")
# new_files.write(files)
# new_files.close()

# 1 - ru, 2 - en, 3 - tr, 4 - ru-en, 5 - ru-tr, 6 - en-tr, 7 - en-ru-tr
# file = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt", "w")
# for i in range (210):
#     # print(i + 1, (i // 30) + 1)
#     file.write(str((i // 30) + 1) + "\n")

import random

file_txt = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt").read().split("\n")
file_ans = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt").read().split("\n")
new_txt = []
new_ans = []


for i in range(210):
    number = random.randint(0, 209 - i)
    new_txt.append(file_txt[number])
    new_ans.append(file_ans[number])
    file_txt = file_txt[:number] + file_txt[number + 1:]
    file_ans = file_ans[:number] + file_ans[number + 1:]
file1 = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt", "w")
file2 = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt", "w")
for i in range(210):
    file1.write(new_txt[i] + "\n")
    file2.write(new_ans[i] + "\n")
file1.close()
file2.close()