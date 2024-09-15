# Hind4sight-Net: Unsupervised Learning for Object Dynamics Prediction

## Project Overview
This project focuses on implementing a joint forward and inverse dynamics model as part of the **Hind4sight-Net** architecture. The model is designed to predict the dynamics of a robot's physical interaction with objects in an unsupervised manner, specifically through poking actions. 

### Key Features:
- **Forward Dynamics Model**: Predicts the future state of the scene, decomposing it into salient object parts and estimating their 3D motion under applied actions.
- **Inverse Dynamics Model**: Infers the actions responsible for observed state transitions between two consecutive point clouds.
- **Unsupervised Learning**: The model segments the scene and predicts 3D motion without requiring labeled data.

## Methodology
The project implements a joint forward and inverse dynamics architecture to model object-centric motion and action inference. The **Forward Dynamics Model** estimates rigid body transformations (rotation and translation) to predict future point clouds, while the **Inverse Dynamics Model** uses consecutive point clouds to infer the actions taken.

## Loss Functions
- **Chamfer Loss**: Ensures geometric consistency between predicted and target point clouds.
- **MSE (Image-to-Image Loss)**: Measures the difference between the predicted and ground-truth 2D image projections.
- **Cross Entropy Loss**: Ensures the accuracy of action predictions in the inverse dynamics model.

The total loss function is a weighted combination of these three losses:
\[
L = \lambda_1 L_{\text{CD}} + \lambda_2 L_{\text{MSE}} + \lambda_3 L_{\text{CE}}
\]
with \( \lambda_1 = 10^5 \), \( \lambda_2 = 10^3 \), and \( \lambda_3 = 1 \).

## Results
- **Forward Dynamics Model**: Successfully predicts future point clouds based on the current state and applied action.
- **Inverse Dynamics Model**: Accurately predicts the action that caused a transition between consecutive point clouds.

For example, the inverse model predicts the following action vector for a set of consecutive point clouds:
\[
\text{Predicted Action} = [-0.0603, 0.1531, -0.0364, -0.0583]
\]
which corresponds to the start and end positions of the poke action.

## Contributors
- **Forward Dynamics Model**: Alisha Syed Karimulla
- **Inverse Dynamics Model**: Oviya Rajavel

## GitHub Repository
Access the full project on GitHub: [CORO Project GitHub Link](https://github.com/zaara08/CORO_Project.git)

## Dataset
The dataset used in this project is publicly available from the paper "Hindsight for Foresight: Unsupervised Structured Dynamics Models from Physical Interaction."

## References
1. Byravan, A., et al. "Se3-pose-nets: Structured deep dynamics models for visuomotor planning and control." arXiv preprint arXiv:1710.00489 (2017).
2. Byravan, A., & Fox, D. "SE3-nets: Learning rigid body motion using deep neural networks." 2017 IEEE International Conference on Robotics and Automation (ICRA), Singapore, 2017.
3. Nematollahi, I., et al. "Hindsight for foresight: Unsupervised structured dynamics models from physical interaction." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2020.
4. The encoder,decoder and the SE3 transformationa are from the github page:[SE3netpose_nets](https://github.com/msieb1/se3net.git)
