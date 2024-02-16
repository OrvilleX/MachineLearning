# 使用方式

本文将主要介绍围绕各个版本的yolo框架的使用方式，包括模型的训练，验证评估以及实际的推理使用方式，其中包括官方
的版本以及其他衍生的版本均会记录在本文档中。  

## 一、Yolov8使用

### 1.1 模型训练

对于不改变源码网络的情况下可以通过安装对应的库直接实现模型的训练、验证与推理，首先我们通过下面的
命令安装对应的依赖库，这里针对pytorch与cuda的安装不多做介绍，直接安装核心的库。

```bash
pip install ultralytics
```

安装好上述依赖的类库后我们通过下方的方式配置对应的训练参数配置文件，首先我们需要通过`yolo copy-cfg`生成一个默认的配置文件`default-copy.yaml`
接着调整其中我们需要关心的以下几个核心参数，然后我们就可以开始训练模型了。

* model: 需要依赖的基础的模型，默认为yolov8n，可以选择其他的模型进行训练；  
* data: 需要训练的数据的文件，需要对应的yaml文件，这点类似yolov5版本的训练参数配置文件； 
* epochs: 训练迭代的次数; 
* batch: 每次训练的批量数据量，需要结合已有的显存决定具体的大小; 
* imgsz: 输入图片的齿轮，可以为单一的数字也是是数组决定宽度与高度; 

对于其他的参数可以从其他渠道进行学习，根据自己的模型进行优化调整。完成上述的参数配置后我们就可以进行模型的训练，通过下方的指令我们采用上面的
配置文件进行模型的训练。  

```bash
yolo cfg=default-copy.yaml
```

### 1.2 模型推理

完成上述对应的模型训练后，我们将得到一个最优的模型来供我们使用，为了能够实际的验证模型的有效性，我们可以通过两种方式来进行推理，可以通过单一图片
或批量将对应文件夹下的图片进行推理，输出对应的推理结果或直接绘制到的对应的图片上并输出，形成较直观的图片结果。  

下面我们以官方的yolov8n模型为例，通过最普通的方式推理一个本地图片: 

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from IPython.display import Image

# 单一图片的推理
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    device="cuda:0"
)
# 常规推理
result = get_prediction("test.jpg", detection_model)
result.export_visuals(export_dir="runs/predict/")
Image("runs/predict/prediction_visual.png")
```

上面是最常规的推理一个图片的方式，但是针对超高像素的图片我们使用切片的方式进行推理，我们将上面的推理的方式进行调整，从而
得出以下新的方式来进行本地图片的推理，修改后的代码如下: 

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 单一图片的推理
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    device="cuda:0"
)

# 切片推理
result = get_sliced_prediction(
    "test.jpg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
# 支持转换到其他的标注数据格式
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
```

如果我们希望针对一个文件夹下的图片都进行推理并输出到一个文件夹下，便于我们查看实际的推理的效果，我们可以采用下述的
代码针对指定文件夹下的图片进行批量的处理，并将结果输出到指定的文件夹下，哭啼的代码例子如下: 

```python
from sahi.predict import predict

# 批量图片推理
result = predict(
    model_type="yolov8",
    model_path="/root/runs/detect/train8/weights/best.pt",
    model_device="cuda:0",
    model_confidence_threshold=0.4,
    source="/root/autodl-tmp/hszb/vest/val/images",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
result.export_visuals(export_dir="/roots/runs/predict/")
```  

至此我们完成了我们训练的yolov8模型的推理，剩下就可以直接运用到生产环境中了。  

