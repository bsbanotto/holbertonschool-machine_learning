import sys
sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/supervised_learning/0x00-object_detection')

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../../data/yolo.h5', '../../data/coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('../../data/yolo')
    for i, name in enumerate(image_paths):
        if "dog.jpg" in name:
            ind = i
            break
    print(image_paths[ind])
    print(predictions[ind])