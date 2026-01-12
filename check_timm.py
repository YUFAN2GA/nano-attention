import timm
print("Timm version:", timm.__version__)
models = timm.list_models('*dino*')
print("DINO models:", models)
