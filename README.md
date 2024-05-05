# closed-symmetry-loop


## Citation
J. Richter-Klug and U. Frese, "Handling Object Symmetries in CNN-based Pose Estimation," 2021 IEEE International Conference on Robotics and Automation (ICRA), Xi'an, China, 2021, pp. 13850-13856, doi: 10.1109/ICRA48506.2021.9561237.

### Abstract
In this paper, we investigate the problems that Convolutional Neural Networks (CNN)-based pose estimators have with symmetric objects. We consider the value of the CNN’s output representation when continuously rotating the object and find that it has to form a closed loop after each step of symmetry. Otherwise, the CNN (which is itself a continuous function) has to replicate an uncontinuous function. On a 1-DOF toy example, we show that commonly used representations do not fulfill this demand and analyze the problems caused thereby. In particular, we find that the popular min-over-symmetries approach for creating a symmetry-aware loss tends not to work well with gradient-based optimization, i.e., deep learning. We propose a representation called "closed symmetry loop" (csl) from these insights, where the angle of relevant vectors is multiplied by the symmetry order and then generalize it to 6-DOF. The representation extends our algorithm from [1] including a method to disambiguate symmetric equivalents during the final pose estimation. The algorithm handles continuous rotational symmetry (e.g., a bottle) and discrete rotational symmetry (e.g., a 4-fold symmetric box). It is evaluated on the T-LESS dataset, where it reaches state-of-the-art for unrefining RGB-based methods.guate symmetric equivalents during the final pose estimation. The algorithm handles continuous rotational symmetry (e.g. a bottle) and discrete rotational symmetry (e.g. a 4-fold symmetric box). It is evaluated on the T-LESS dataset, where it reaches state-of-the-art for unrefining RGB-based methods.

### Keywords
Automation, Conferences, Pose estimation, Toy manufacturing industry, Convolutional neural networks, Optimization

### URL
[Link to the paper at IEEExplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561237&isnumber=9560666)

---

## Docker

To build and run the Docker container for the "closed-symmetry-loop" project, follow these steps:

1. Open a terminal and navigate to the project directory:
   ```bash
   cd /home/domin/Documents/GitHub/closed-symmetry-loop
   ```

2. Build the Docker image using the following command:
   ```bash
   sudo docker build . -t closed-symmetry-loop
   ```

3. Once the image is built, you can run the container with the following command:
   ```bash
   sudo docker run -it --rm --gpus all -v /home/domin/Documents/GitHub/closed-symmetry-loop/learning.ipynb:/tf/notebooks -p 8888:8888 --name robot_vision24 closed-symmetry-loop
   ```

Alternatively, you can use the `docker-compose.yml` file to simplify the process:

1. Open a terminal and navigate to the project directory:
   ```bash
   cd /home/domin/Documents/GitHub/closed-symmetry-loop/
   ```

2. Start the Docker container using the following command:
   ```bash
   sudo docker compose -f "docker-compose.yml" up --build
   ```

These commands will build and start the Docker container, allowing you to run your project in a consistent and isolated environment.

---

