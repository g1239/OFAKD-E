import timm
#列举当前timm版本支持的backbone
model_names = timm.list_models('*vit*',pretrained= True)
for i in model_names:
    print(i)