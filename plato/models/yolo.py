"""The YOLOV5 model for PyTorch."""
# from yolov5.models import yolo
# from yolov5.utils.torch_utils import time_synchronized
from packages.yolov5.yolov5.models import yolo
from packages.yolov5.yolov5.utils.torch_utils import time_synchronized

from plato.config import Config

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Model(yolo.Model):
    """The YOLOV5 model with cut layer support."""
    def __init__(self, model_config, num_classes):
        super().__init__(cfg=model_config, ch=3, nc=num_classes)
        Config().params['grid_size'] = int(self.stride.max())

    def forward_to(self, x, cut_layer=4, profile=False):
        y, dt = [], []  # outputs

        for m in self.model:
            if m.i == cut_layer:
                return x

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j]
                                    for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(
                    x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def forward_from(self, x, cut_layer=4, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.i < cut_layer:
                y.append(None)
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j]
                                    for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(
                    x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    @staticmethod
    def get_model():
        """Obtaining an instance of this model provided that the name is valid."""
        if hasattr(Config().trainer, 'model_config'):
            return Model(Config().trainer.model_config,
                         Config().data.num_classes)
        else:
            # return Model('yolov5s.yaml', Config().data.num_classes)
            return Model('/home/ubuntu/Pisces/packages/yolov5/yolov5/models/yolov5s.yaml', Config().data.num_classes)
            ######## 下面是自己猜的，使用预训练模型
            # return self.model.load_state_dict(torch.load(model_path))
            # model = Model('/home/ubuntu/Pisces/packages/yolov5/yolov5/models/yolov5s.yaml', Config().data.num_classes)
            # import torch
            # model=torch.load(my_model.pth)

            # import torch
            # ret_model = Model('/home/ubuntu/Pisces/packages/yolov5/yolov5/models/yolov5s.yaml', Config().data.num_classes)
            # best_model_path = "/home/ubuntu/Pisces/plato/models/pretrained_model/best.pt"
            # best_model = torch.load(best_model_path)
            # print(type(best_model))
            # print(best_model)
            # best_model_parameters = best_model.state_dict()
            # ret_model.load_state_dict(best_model_parameters)
            # return ret_model

            # import torch
            # import torch.nn as nn
            # l = nn.Linear(3, 5) # 为什么是3，5
            # w = list(l.parameters())
            # model = Model('/home/ubuntu/Pisces/packages/yolov5/yolov5/models/yolov5s.yaml', Config().data.num_classes)
            # model.load_state_dict(w)
            # return model