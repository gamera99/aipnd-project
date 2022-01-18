# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Nelson Walker

01/17/2022

I would like to thank my mentor Mr. Himanshu M
who helped me troubleshoot my final project Part 2 (train.py and predict.py)

My final project consists of these files:

- train.py
   - This trains a network using either pretrained networks densenet121 or vgg16.
   - It trains the network and saves a checkpoint file for use in predict.py.
   - Sample command line is:
   - python train.py --arch "densenet121" --hidden_units 512 --epochs 5 --gpu "True"

- predict.py
   - This predicts the highest probability flower names for a test image.
   - It uses a saved checkpoint file that was created in train.py above.
   - Sample command line is:
   - python predict.py --image "flowers/test/50/image_06320.jpg" --gpu=True

- test_checkpoint.py
    - I built a new executable called test_checkpoint.py that tests my saved checkpoint file against the test database,
    at the suggestion of my mentor Mr. Himanshu M.

My project contains screenshots of logging output for train.py, predict.py, and test_checkpoint.py.

# Notes
- In my Part 1 Jupyter NB ipynb, I was able to train my network to over 83% accuracy on the validation dataset.
- In my Part 2 train.py, I used my same code but was unable to train to much better than 53% accuracy.
- I changed many parameters, learning rates, added dropouts etc.
- I built a new executable called test_checkpoint.py that tested my saved checkpoint file against the test database,
at the suggestion of my mentor Mr. Himanshu M.


I learned a lot in this Udacity course from the professors, my mentor Mr. Himanshu M.  When I was having trouble I analyzed some github final projects and learned new ideas for configuring my project.

Thank you very much

Nelson Walker

01/17/2022

Udacity Curriculum:  AI Programming with Python
