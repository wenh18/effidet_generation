import torch
import torch.nn as nn
import torch.nn.functional as F


def get_TGNetwork( d_model=100, gate_len=34, select_embbed_len=128, kernel_number=512):
    model = TGNetwork(d_model, gate_len, select_embbed_len, kernel_number, )
    return model


class TGNetwork(nn.Module):
    def __init__(self, d_model, gate_len=0, select_embbed_len=0, kernel_number=0):
        super(TGNetwork, self).__init__()
        self.gate_len = gate_len
        self.select_embed_len = select_embbed_len




        self.TaskLinear = nn.Sequential(
            nn.Linear(d_model, gate_len * 8, bias=False),
            nn.BatchNorm1d(gate_len * 8),
            nn.ReLU(),
            nn.Linear(gate_len * 8, gate_len * 32, bias=False),
            nn.BatchNorm1d(gate_len * 32),
            nn.ReLU(),
            nn.Linear(gate_len * 32, gate_len * 128, bias=False),
            nn.BatchNorm1d(gate_len * 128),
            nn.ReLU(),
            nn.Linear(gate_len * 128, gate_len * 256, bias=False),
            nn.BatchNorm1d(gate_len * 256),
            nn.ReLU(),
            nn.Linear(gate_len * 256, gate_len * 512, bias=False),
            nn.BatchNorm1d(gate_len * 512),
            nn.ReLU(),
            nn.Linear(gate_len * 512, gate_len * select_embbed_len, bias=False),
            nn.BatchNorm1d(gate_len * select_embbed_len),
            nn.ReLU(),

        )

        self.LayerGating = nn.Sequential(
            nn.Linear(select_embbed_len, 4 * select_embbed_len, bias=False),
            nn.ReLU(),
            nn.Linear(4 * select_embbed_len, 8 * select_embbed_len, bias=False),
            nn.ReLU(),
            nn.Linear(8 * select_embbed_len, 16 * select_embbed_len, bias=False),
            nn.ReLU(),
            nn.Linear(16 * select_embbed_len, 2 * kernel_number, bias=False),
            nn.ReLU(),
            nn.Linear(2 * kernel_number, kernel_number, bias=False))


    def forward(self, prompt, ):
        # prompt:[batchsize,prompt_len]

        task_cls = prompt


        layer_encoding = self.TaskLinear(task_cls)

        layer_encoding = layer_encoding.view(-1, self.gate_len, self.select_embed_len)


        # selection_embbeding:[batchsize,layer_len,select_layer_encoding_len]

        layer_selection = self.LayerGating(layer_encoding)

        layer_selection = F.sigmoid(layer_selection)
        layer_selection = torch.clamp(1.2 * layer_selection - 0.1, min=0, max=1)
        discrete_gate = StepFunction.apply(layer_selection)

        if self.training:
            discrete_prob = 0.5
            mix_mask = layer_selection.new_empty(
                size=[layer_selection.size()[0], self.gate_len, 1]).uniform_() < discrete_prob
            layer_selection = torch.where(mix_mask, discrete_gate, layer_selection)
        else:
            layer_selection = discrete_gate

        return layer_selection


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theshold=0.49999):
        return (x > theshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None