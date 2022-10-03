import torch
import os
import warnings
import models.common


def f2():
    print('---------------------load model...')
    path = os.getcwd() + '/../'
    model_path = os.getcwd() + '/../yolov5s.pt'
    model = torch.hub.load(path, 'custom', path=model_path, source="local")

    if not isinstance(model, models.common.AutoShape):
        warnings.warn(f'nodel type error {type(model)}')
        return

        # or file, Path, PIL, OpenCV, numpy, list
    imgs = [
        os.getcwd() + '/../data/images/dog.jpg',
        os.getcwd() + '/../data/images/bus.jpg'
    ]

    print('---------------------start ...')

    for img in imgs:
        # Inference
        results = model(img)
        if not isinstance(results, models.common.Detections):
            warnings.warn(f'nodel type error {type(results)}')
            continue

        print(f'\n---------------------------- {img}')

        # Results
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        result_object = results.crop(save=False)

        for i, obj in enumerate(result_object):
            idx = int(obj['cls'])
            label = results.names[idx]
            conf = float(obj['conf'])
            box = obj['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            print(f'{i + 1}. 名称: {label:10} 可信度: {conf:.2f} 位置: ({x1:3},{y1:3}) ({x2:3},{y2:3})')


f2()
