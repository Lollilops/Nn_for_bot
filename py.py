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
#1 - tr,2 - ru, 3 - en
import random
file_txt = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/Zhenya.txt").read().split("\n"))[:-1]
# print(file_txt)
# file_ans = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt").read().split("\n")
new_ind = list()
new_file_txt = list()
for ind in file_txt:
    new_ind.append(int(ind[0]))
    new_file_txt.append(ind[2:])
new_txt = []
new_ans = []
file_txt = new_file_txt
file_ans = new_ind
for i in range(106):
    number = random.randint(0, 106 - 1 - i)
    if file_ans[number] == 1:
        lang = "tr"
    elif file_ans[number] == 2:
        lang = "ru"
    elif file_ans[number] == 3:
        lang = "en"
    new_txt.append(file_txt[number] + "," + lang)
    # new_ans.append(file_ans[number])
    file_txt = file_txt[:number] + file_txt[number + 1:]
    file_ans = file_ans[:number] + file_ans[number + 1:]
file1 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt", "w+")
# file2 = open("/Users/yurygornostaev/VisualcodeProjects /LAES/data_txt/trn/txt_ans.txt", "w")
for i in range(231):
    file1.write(new_txt[i] + "\n")
    # file2.write(new_ans[i] + "\n")
file1.close()
# file2.close()

# ----------------
# file_v = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/Zhenya.txt").read().split("\n"))[:-1]
# file_2 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/Zhenya.txt", "w+")
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
# ------------
# file_v = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/1.txt").read().split("\n"))
# file_2 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/2.txt", "w+")
# file_3 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/3.txt", "w+")
# # print(file_v)
# for i in file_v:
#     num = i.index("\t")
#     file_3.write(i[:num] + "\n")
#     file_2.write(i[num+1:] + "\n")
# file_2.close()
# file_3.close()
#---------------
# file_v = (open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/Zhenya.txt").read().split("\n"))[:-1]
# file_2 = open("/Users/yurygornostaev/VisualcodeProjects /Nn_for_bot/Nn_for_bot/data_txt/trn/Zhenya.txt", "w+")
# for i in file_v:
#     if int(i[0]) > 3:
#         break
#     else:
#         file_2.write(i)
# file_2.close()
