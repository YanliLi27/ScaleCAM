import os
import pandas as pd

path1 = 'D:\\ESMIRA\\ESMIRA_ultitest\\CSA_Wrist_TRA'

list_csa = os.listdir(path1)
list_shorten_csa = [item.split('-')[2] for item in list_csa]
save1 = 'C:\\Users\\yli5\\Desktop\\im_6_11\\CSAintest.csv'
csa_dict = {'CSA_intest':list_csa, 'CSA_id':list_shorten_csa}
df_csa = pd.DataFrame(csa_dict)
df_csa.to_csv(save1)

print(list_csa)

path2 = 'D:\\ESMIRA\\ESMIRA_ultitest\\EAC_Foot_COR'

list_eac = os.listdir(path2)
list_shorten_eac = [item.split('-')[2] for item in list_eac]
print(list_eac)
save2 = 'C:\\Users\\yli5\\Desktop\\im_6_11\\EACintest.csv'
eac_dict = {'EAC_intest':list_eac, 'EAC_id':list_shorten_eac}
df_eac = pd.DataFrame(eac_dict)
df_eac.to_csv(save2)