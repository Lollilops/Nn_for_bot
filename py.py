# files = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt").read()
# files = files.replace("/", "")
# # files = files.replace(")", "")
# new_files = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt.txt", "w")
# new_files.write(files)
# new_files.close()
# ---------------
# 1 - ru, 2 - en, 3 - tr, 4 - ru-en, 5 - ru-tr, 6 - en-tr, 7 - en-ru-tr
# file = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt", "w")
# for i in range (210):
#     # print(i + 1, (i // 30) + 1)
#     file.write(str((i // 30) + 1) + "\n")
# ---------------
# import random
# file_txt = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt").read().split("\n")
# file_ans = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt").read().split("\n")
# new_txt = []
# new_ans = []

# for i in range(231):
#     number = random.randint(0, 230 - i)
#     new_txt.append(file_txt[number])
#     new_ans.append(file_ans[number])
#     file_txt = file_txt[:number] + file_txt[number + 1:]
#     file_ans = file_ans[:number] + file_ans[number + 1:]
# file1 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt", "w")
# file2 = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt", "w")
# for i in range(231):
#     file1.write(new_txt[i] + "\n")
#     file2.write(new_ans[i] + "\n")
# file1.close()
# file2.close()

# ----------------
# file_v = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt").read().split("\n"))[:-1]
# file_2 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt", "w+")
# # file_3 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/3.txt", "w+")
# new_file_v = []
# new_file_ind = []
# # print(file_v)
# # k = 5
# file_ind = []
# for i in file_v:
#     num = i.index("\t")
#     file_ind.append(int(i[:num]))
#     # print(i[:num])
#     # print(i[num + 2:])
# for k in range(1, 8):
#     ind_flag = True
#     while ind_flag:
#         if file_ind.count(k) != 0:
#             num = file_ind.index(k)
#             # print(num)
#             file_ind = file_ind[:num] + file_ind[num+1:]
#             new_file_ind.append(k)
#             new_file_v.append(file_v[num])
#             file_v = file_v[:num] + file_v[num + 1:]
#         else:
#             ind_flag = False
# for i in range(len(new_file_ind)):
#     file_2.write(new_file_v[i] + "\n")
# file_2.close()
file_v = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/1.txt").read().split("\n"))
file_2 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt", "w+")
file_3 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/3.txt", "w+")
# print(file_v)
for i in file_v:
    num = i.index("\t")
    file_3.write(i[:num] + "\n")
    file_2.write(i[num+1:] + "\n")
file_2.close()
file_3.close()