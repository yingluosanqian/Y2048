import torch

const_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
const_device_cpu = "cpu"
const_device_gpu = "cuda"
const_boarder_size = 3

const_action = ["Left", "Right", "Up", "Down"]
