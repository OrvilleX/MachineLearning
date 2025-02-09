import os
import argparse


if __name__ == '__main__':
    """
    通过本工具可以根据负向样本图片生成对应的空标注文件
    -i (--image)代表需要生成空txt的图片文件夹路径
    -o (--output)代表生成的txt文件保存的文件夹路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default='../images/', help='images path')
    parser.add_argument('-o', '--output', type=str, default='../labels/', help='labels path')
    opt = parser.parse_args()
    image = opt.image
    output = opt.output
    for i, j, k in os.walk(image):
        for s in k:
            s = s.strip('.jpg')
            full_path = output + s + '.txt'
            file = open(full_path, 'w')
            file.close()
            print(s)
