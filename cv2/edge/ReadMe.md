# 边缘检测

存在有些场景需要根据图像识别对应的边缘特征，为了能够对比并体现各种边缘检测算法的区别，本项目主要将opencv中常用的几种边缘检测算法进行介绍。通过本代码可以将输入的图片显示出不同的边缘检测后结果的图片，同时也可以支持批量将图片转换为多个边缘检测结果的图片进行存储。

---

## 一、主要结构

本项目中主要存在2个核心函数，其中`apply_edge_detection`是内置了多个边缘检测算法函数，实现将输入的图片进行边缘检测并输出，对应下面的介绍也主要集中针对该函数中各类边缘检测算法的使用以及应用场景；同时对应的`display_results`函数作用是将上述函数的输出进行可视化，但是其函数就只能支持单个图片进行检测并输出，从而能够直观的感受到不同边缘检测算法的识别差异。

---

## 二、边缘检测识别算法介绍

### 2.1 `auto_canny`

- 用法示例:

  ```python
  edges = auto_canny(gray_image, sigma=0.33)
  ```

- 参数说明:
  - `image`: 输入灰度图像。
  - `sigma` (可选): 控制低高阈值设定范围的参数，默认值为 0.33。较大的 sigma 会使得低、高阈值之间的差距更大。

- 功能描述:
  - 根据输入图像的中值自动计算 Canny 算法的低、高阈值，适用于光照或对比度不均的图像。

- 适用场景:
  - 在图像亮度变化较大或对比度不均的情况下能够自适应调整，提升边缘检测的鲁棒性。

### 2.2 `Canny` 边缘检测

- 用法示例:

  ```python
  canny_edges = cv2.Canny(blurred, 30, 120)
  ```

- 参数说明:
  - `blurred`: 预处理（如高斯模糊或直接使用灰度图）的输入图像。
  - `30` 和 `120`: 分别为低阈值和高阈值，用于判断边缘强弱。

- 功能描述:
  - 通过计算图像梯度来检测边缘，能够有效捕捉到图像的明显边缘信息。

- 适用场景:
  - 图像噪声较低且光照均衡的场景，非常适合进行基于固定阈值的边缘检测。

### 2.3 `Sobel` 边缘检测

- 用法示例:

  ```python
  sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
  sobel_edges = np.uint8(cv2.magnitude(sobelx, sobely))
  ```

- 参数说明:
  - `cv2.CV_64F`: 指定输出图像深度以捕捉负梯度。
  - `1, 0` 或 `0, 1`: 分别计算 x 或 y 方向的梯度。
  - `ksize=3`: 核大小，影响检测的敏感性和噪声平滑。

- 功能描述:
  - 通过计算图像各方向梯度的幅值来检测边缘，对噪声较为敏感但能明确展示梯度变化。

- 适用场景:
  - 当需要获取图像梯度信息以辅助后续处理（如方向估计）时，Sobel 算法表现较好。

### 2.4 `Laplacian` 边缘检测

- 用法示例:

  ```python
  laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
  laplacian_edges = np.uint8(np.absolute(laplacian))
  ```

- 参数说明:
  - `cv2.CV_64F`: 输出图像的深度，用于获取更精确的梯度值。
  - 使用 `np.absolute` 将负值转换为正，并转换为 8 位图像。

- 功能描述:
  - 利用二阶导数来检测边缘，能够捕捉到快速变化的灰度信息。

- 适用场景:
  - 对于边缘不明显或细节丰富的图像，Laplacian 算法可以提供补充信息，特别是在噪声较低时效果更佳。

### 2.5 `Scharr` 边缘检测

- 用法示例:

  ```python
  scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
  scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
  scharr_edges = np.uint8(cv2.magnitude(scharrx, scharry))
  ```

- 参数说明:
  - 类似于 Sobel，但 Scharr 专为更好地检测小细节而设计。

- 功能描述:
  - 在较小的卷积核下能够更准确地捕捉到细节边缘，适用于细微结构的检测。

- 适用场景:
  - 当处理高分辨率或细节丰富的图像时，Scharr 算法能提供比 Sobel 更高的精度。

### 2.6 `Prewitt` 边缘检测

- 用法示例:

  ```python
  kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
  kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
  prewittx = cv2.filter2D(blurred, cv2.CV_64F, kernelx)
  prewitty = cv2.filter2D(blurred, cv2.CV_64F, kernely)
  prewitt_edges = np.uint8(cv2.magnitude(prewittx, prewitty))
  ```

- 参数说明:
  - `kernelx` 和 `kernely`: 自定义的卷积核，用于分别提取横向和纵向的边缘信息。
  - 使用 `cv2.filter2D` 进行卷积操作，再通过计算梯度幅值得到边缘图像。

- 功能描述:
  - 虽然 OpenCV 没有直接提供 Prewitt 算法，但通过自定义卷积核可以实现类似效果，适用于粗略的边缘检测。

- 适用场景:
  - 在对计算速度要求较高且对细节要求不太严格的场景下，Prewitt 算法提供了一个简单高效的选择。

### 2.7 `融合边缘检测（Fused Edges）`

- 用法示例:

  ```python
  # 过滤处理后的 Canny 和 Prewitt 结果
  _, canny_filtered = cv2.threshold(canny_edges, 40, 255, cv2.THRESH_TOZERO)
  _, prewitt_filtered = cv2.threshold(sobel_edges, 10, 255, cv2.THRESH_TOZERO)
  prewitt_unique = cv2.subtract(prewitt_filtered, canny_filtered)
  alpha = 0.8
  beta = 1.0
  fused_edges = cv2.addWeighted(canny_filtered, beta, prewitt_unique, alpha, 0)
  ```

- 参数说明:
  - `alpha` 和 `beta`: 分别代表 Prewitt 独有部分与 Canny 结果的权重比例。
  - 阈值操作用于提取主要边缘信息，再通过加权融合得到增强后的边缘图像。

- 功能描述:
  - 通过融合 Canny 与 Prewitt（或 Sobel 变种）的检测结果，综合利用各自的优势，进而获得更为鲁棒和全面的边缘信息。

- 适用场景:
  - 在多种边缘检测方法各有优劣的情况下，融合方法可适用于复杂场景，提升整体边缘检测性能。

---

## 3. 边缘检测的最佳实践

以下是使用这些边缘检测算法时的一些建议和注意事项：

### 3.1 图像预处理
- **灰度转换**：  
  所有边缘检测算法推荐先将输入图像转换为灰度图，这样能够简化计算并提高算法效率。
  
- **噪声抑制**：  
  为了减少图像中噪声对边缘检测结果的干扰，可使用高斯模糊或者其他降噪方法。示例代码中已提到高斯模糊步骤，但默认被注释；如遇噪声问题，请适当启用并调整参数。

### 3.2 多算法融合
- **加权融合**：  
  示例中将 Canny 结果与经过阈值处理过的 Prewitt 结果进行融合（通过 `cv2.addWeighted`），可有效综合不同算法的优点。  
- **权重调节**：  
  根据不同图像场景，可试着调整融合权重参数 `alpha`（Prewitt 权重）与 `beta`（Canny 权重），以达到最佳效果。

---
