# Multi-Task-Vision-System-Simultaneous-Object-Detection-and-Semantic-Segmentation
This project demonstrates a lightweight deep learning network capable of performing object detection and semantic segmentation simultaneously. Built using MobileNetV2 and TensorFlow, the model is optimized for real-time applications on mobile devices. It employs SSDLite for object detection and DeepLabV3+ for semantic segmentation.

### The Goal

I wanted to create a network that could do two things:
1. **Object Detection**: Detect and label objects in images using SSD (Single Shot MultiBox Detector).
2. **Semantic Segmentation**: Identify and segment different parts of an image using DeepLabV3+.

To keep things light and fast, I used MobileNetV2. I've used deptwhise separable convolutions when implementing SSD, so technically I've implemented what's called SSDLite. However it's exactly the same as SSD, just switching standard convolutions operations with depthwise separable convolutions.

### Key Highlights

- **Built from Scratch**: Everything, including the backbone networks and detection/segmentation heads, were coded from scratch using NumPy and TensorFlow.
- **Compact and Efficient**: The final network has around 4M parameters, which is pretty small and efficient for real-time tasks.
- **Training**: Trained the models on a robot warehouse dataset with three classes: people, forklifts, and rails.

### Results

The models didn’t break any records, but considering the small dataset and the fact that everything was built and trained from scratch, the results are pretty cool! It shows that even with limited data, these computer vision techniques can be quite effective.

### What I Learned

This project was.. challenging! I learned a lot about building and training models from scratch and got a better understanding of how to make efficient, real-time computer vision solutions. It wasn't easy but seeing decent boxes and segmentation masks coming out from the network was super rewarding.

## Why What’s inside this Repo

I hope this repo can help other people getting to know easier architectures for performing object detection and semantic segmentation

Here's a quick breakdown of the repo: 

- **`/data`**: this folder contains metadata for training, validation and test, but unfortunately the dataset it's proprietary and I cannot share it outside my company.
- **`/models`**: this folder contains trained model, I shared MobileNetV2 with SSDLite and DeepLabV3+ trained for 105 epochs.
- **`/ssdseglib`**: this is a custom python module were all the code for this project it's organized, the naming of the files should be self-explanatory and there are type hints, docstrings and comments for helping having an easier understanding.
- **`01-ssd-framework-single-shot-detector-for-object-detection.ipynb`**: in this notebook I explain the core idea behind SSD framework for object detection.
- **`02-data-encoding-and-decoding.ipynb`**: in this notebook I explain the data formats expected by networks using SSD for object detection.
- **`03-multi-task-network-ssdlite-deeplabv3plus-training.ipynb`**: in this notebook I run the main experiment, so you'll find data loading, model training and results.


## Conclusion

This project was a great learning experience and a fun challenge. It’s a good example of how you can handle object detection and segmentation together with a small, efficient network. Hope you find it interesting too!

## Future work

In multitask learning, balancing the loss functions of different tasks can significantly impact performance. A promising approach for future work is to implement **uncertainty weighting**, where each task's loss is dynamically weighted based on its inherent uncertainty. This method adapts the model to focus more on tasks with less certain predictions, improving overall performance. Here's how you could approach this:

1. **Incorporate Task Uncertainty into Loss Weighting**:
   - Modify the multitask loss function to include learnable uncertainty parameters for each task. These parameters adjust the weighting dynamically during training.
   - The formula for the weighted loss can be inspired by Gaussian likelihood-based uncertainty:
     \[
     L_{\text{total}} = \sum_{i=1}^{N} \frac{1}{2\sigma_i^2} L_i + \log(\sigma_i)
     \]
     Here, \(L_i\) is the loss for task \(i\), and \(\sigma_i\) represents the uncertainty for that task. Smaller \(\sigma_i\) values indicate higher confidence in task \(i\), leading to higher weighting.

By integrating uncertainty weighting, the model can prioritize tasks more effectively, making it robust and adaptive to real-world variations and noise.