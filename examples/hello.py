import torch
import os
import cv2


def hell1():
    print('---------------------load model...')

    path = os.getcwd() + '/../'
    model_path = os.getcwd() + '/yolov5s.pt'  # os.getcwd() + '/../yolov5s.pt'

    model = torch.hub.load(path, 'custom', path=model_path, source="local")

    imgs = [
        # r'E:\code\opencv_demo\yolov5\data\images\bus.jpg',
        # r'E:\code\opencv_demo\yolov5\data\images\zidane.jpg',
        r'E:\code\opencv_demo\demo\res\a.png',
        # r'E:\code\opencv_demo\demo\res\b.png',
    ]

    print('---------------------start ...')

    # cap = cv2.VideoCapture(r'F:\Develop\ServerE\bvcr\src\cr\clay\demo\Camera Road 01.avi')  # but your video here

    index = 0
    for img_path in imgs:
        # Inference
        results = model(img_path)
        print(f'\n---------------------------- {img_path}')

        # Results
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        result_object = results.crop(save=False)

        img = cv2.imread(img_path)

        for i, obj in enumerate(result_object):
            idx = int(obj['cls'])
            label = results.names[idx]
            conf = float(obj['conf'])
            box = obj['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            print(f'{i + 1}. 名称: {label:10} 可信度: {conf:.2f} 位置: ({x1:3},{y1:3}) ({x2:3},{y2:3})')

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, "{}({})".format(label, int(conf * 100)), (x1 + 10, y1 + 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (50, 50, 200), 1)

        index = index + 1
        cv2.imshow("img {}".format(index), img)

    cv2.waitKey()


print(torch.cuda.is_available())
hell1()
