import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        # nn.Module.__init__(self)
        # self._tower = nn.Sequential(
        #     nn.Conv2d(6, 64, 3, 1, 1, bias = False),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, 1, 1, bias = False),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 32, 3, 1, 1, bias = False),
        #     nn.ReLU(True),
        #     nn.Flatten()
        # )
        # self._logits = nn.Sequential(
        #     nn.Linear(32 * 4 * 9, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 235)
        # )
        # self._value_branch = nn.Sequential(
        #     nn.Linear(32 * 4 * 9, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 1)
        # )
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

        #在原代码的基础上添加残差块
        nn.Module.__init__(self)
        self._tower1 = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True)
        )
        self.RSblock1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64)
        )
        self.RSblock2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64)
        )
        self.RSblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64)
        )

        self._tower2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower1(obs)
        

        #第一个残差块
        RS = hidden
        hidden = self.RSblock1(hidden)
        hidden += RS 
        hidden = F.relu(hidden)


        #第二个残差块
        RS = hidden
        hidden = self.RSblock2(hidden)
        hidden += RS 
        hidden = F.relu(hidden)


        #第三个残差块
        RS = hidden
        hidden = self.RSblock3(hidden)
        hidden += RS 
        hidden = F.relu(hidden)


        hidden = self._tower2(hidden)


        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value