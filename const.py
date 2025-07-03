import torch

const_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
const_device_cpu = torch.device("cpu")
const_device_gpu = torch.device("cuda")
const_boarder_size = 3

const_action = ["Left", "Right", "Up", "Down"]
const_fix_action_order = [1, 3, 0, 2]
const_num_of_res_tower = 6
