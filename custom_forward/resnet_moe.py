from custom_model import resnet_moe

from .registry import register_method

_target_class = resnet_moe

@register_method
def forward_backbone(self, x, require_route):
    routing_list = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)


    for m in self.modules():
        if hasattr(m, 'get_routing_weights'):
            routing = m.get_routing_weights()
            routing_list.append(routing)

    return (x, routing_list) if require_route else x


@register_method
def forward(self, x, require_route=False):
    if requires_route:
        x, route = self.forward_backbone(x, require_route=True)
        return x, route
    else:
        x = self.forward_backbone(x, require_route=False)                       
        return x                                          





