import os
from PIL import Image
import numpy as np
import tqdm


def main():
    # 数据集通道数
    img_channels = 3
    # 数据集路径
    img_dir = "data/voc_trainval_imgs"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    # 便利数据集路径下 以.jpg为后缀的图片
    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    # 累计mean和std，三个通道，这里是RGB，PIL库中的Image.open 默认RGB，cv2.imread是BGR
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    # 统计数据集长度
    print(f"INFO: {len(img_name_list)} imgs in total")
    for img_name in tqdm.tqdm(img_name_list,total=len(img_name_list)):
        img_path = os.path.join(img_dir, img_name)
        # 对数据集进行归一化
        img = np.array(Image.open(img_path)) / 255.
        # 对每个维度进行统计，Image.open打开的是HWC格式，最后一维是通道数
        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()


