# from mobilenet_1st import mobilenet_1st_floor
from mobilenet_3rd import mobilenet_3rd_floor
from mobilenet_v3_3rd import mobilenet_v3_3rd_floor
from efficientnet_3rd import efficientnet_3rd_floor
# from efficientnet_4th import efficientnet_3rd_floor
from resnet_2nd import resnet_2nd_floor

from mobilenet_v3_1st_rev01 import mobilenet_1st_floor
from mobilenet_v3_2nd import mobilenet_2nd_floor



path = "/mnt/d/burr/make_pth_train_test/A_AA_B_C_D_E_rev01/오분류"
# path = 'dd/'
floor_1= mobilenet_1st_floor(path)
r = [i[1] for i in floor_1]
print(r, floor_1, len(floor_1))
# floor_3= mobilenet_3rd_floor(path)
# print(floor_3)
# floor_3 = mobilenet_2nd_floor(path)
# print(floor_3)
# floor_4= efficientnet_3rd_floor(path)
# print(floor_4, len(floor_4))
# resnet_2nd_floor
# AB= resnet_2nd_floor(path)
# print(AB, len(AB))