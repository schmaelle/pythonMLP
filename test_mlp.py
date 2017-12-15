import numpy as np
from data_generator import data_generator
import cv2
import MLPFunctions as mlpFunc
import time

"""
visualisation and test environment provided from

Prof. Dr. Juergen Brauer University of applied science Kempten

https://github.com/juebrauer/Book_DeepLearningAndBeyond_Exercises


"""

# test data parameters
WINSIZE = 300
NR_CLUSTERS = 5
NR_SAMPLES_TO_GENERATE = 2500

# MLP parameters
NR_EPOCHS = 41
LEARN_RATE = 0.1

# visualization parameters
RADIUS_SAMPLE = 3
COLOR_CLASS0 = (255,0,0)
COLOR_CLASS1 = (0,0,255)
NR_TEST_SAMPLES = 2500

# for saving images
image_counter = 0


def visualize_decision_boundaries(the_mlp, epoch_nr):

    global image_counter

    # 1. generate empty white color image
    img = np.ones((WINSIZE, WINSIZE, 3), np.uint8) * 255

    for i in range(NR_TEST_SAMPLES):

        # 2. generate random coordinate in [0,1) x [0,1)
        rnd_x = np.random.rand()
        rnd_y = np.random.rand()

        # 3. prepare an input vector
        input_vec = np.array( [rnd_x, rnd_y] )

        # 4. now do a feedforward step with that
        #    input vector, i.e., compute the output values
        #    of the MLP for that input vector
        scaleInp = 1#np.amax(input_vec)
        the_mlp.inputVec = input_vec/scaleInp
 
        the_mlp.feedForward()


        # 5. now get the predicted class from the output
        #    vector
        output_vec = the_mlp.outputVec
        class_label = 0 if output_vec[0]>output_vec[1] else 1

        # 6. Map class label to a color
        color = COLOR_CLASS0 if class_label == 0 else COLOR_CLASS1

        # 7. Draw circle
        sample_coord = (int(input_vec[0] * WINSIZE),
                        int(input_vec[1] * WINSIZE))
        cv2.circle(img, sample_coord, RADIUS_SAMPLE, color)

    # 8. show image with decision boundaries
    cv2.rectangle(img, (WINSIZE-120,0), (WINSIZE-1,20),
                  (255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                "epoch #"+str(epoch_nr).zfill(3),
                (WINSIZE - 110, 15), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.imshow('Decision boundaries of trained MLP', img)
    c = cv2.waitKey(1)

    # 9. save that image?
    if False:
        filename = "V:/tmp/img_{0:0>4}".format(image_counter)
        image_counter +=1
        cv2.imwrite(filename + ".png", img)




# 3. generate training data
my_dg = data_generator()
data_samples = \
    my_dg.generate_samples_two_class_problem(NR_CLUSTERS,
                                             NR_SAMPLES_TO_GENERATE)
nr_samples = len(data_samples)


# 4. generate empty image for visualization
#    initialize image with white pixels (255,255,255)
img = np.ones((WINSIZE, WINSIZE, 3), np.uint8) * 255


# 5. visualize positions of samples
for i in range(nr_samples):

    # 5.1 get the next data sample
    next_sample = data_samples[i]

    # 5.2 get input and output vector
    #     (which are both NumPy arrays)
    input_vec = next_sample[0]
    output_vec = next_sample[1]

    # 5.3 prepare a tupel from the NumPy input vector
    sample_coord = (int(input_vec[0]*WINSIZE),
                    int(input_vec[1]*WINSIZE))

    # 5.4 get class label from output vector
    if output_vec[0]>output_vec[1]:
        class_label = 0
    else:
        class_label = 1
    color = (0,0,0)
    if class_label == 0:
        color = COLOR_CLASS0
    elif class_label == 1:
        color = COLOR_CLASS1

    # 5.5
    cv2.circle(img, sample_coord, RADIUS_SAMPLE, color)


# 6. show visualization of samples
cv2.imshow('Training data', img)
c = cv2.waitKey(1)
#cv2.imwrite("V:/tmp/training_data.png", img)
#cv2.destroyAllWindows()

# 1. create a new MLP
myMLP = mlpFunc.MLP(input_vec)

mlpStruct = [2, 10, 6, 2]
mlpTf = ["nothing", "tanh", "tanh", "nothing"]
 
# 2. build a MLP
for neuInLay, tfInLay, layerNr in zip(mlpStruct, mlpTf, range(0,len(mlpStruct))):
    myMLP.add(neuInLay, layerNr, tfInLay)


# 7. now train the MLP
for epoch_nr in range(NR_EPOCHS):

    print("Training epoch#", epoch_nr)

    sample_indices = np.arange(nr_samples)
    np.random.shuffle(sample_indices)

    start = time.time()
    for train_sample_nr in range(nr_samples):

        # get index of next training sample
        index = sample_indices[train_sample_nr]

        # get that training sample
        next_sample = data_samples[index]

        # get input and output vector
        # (which are both NumPy arrays)
        input_vec = next_sample[0]
        output_vec = next_sample[1]

        # train the MLP with that vector pair
        scaleInp = 1#np.amax(input_vec)
        myMLP.inputVec = input_vec/scaleInp
 
        myMLP.feedForward()

        # > calculates new weights, depending on learn rate
        myMLP.backprop(output_vec)
    ende = time.time()
    print("time = %f" % (ende-start) )


    visualize_decision_boundaries(myMLP, epoch_nr)
    #input("Press Enter to train next epoch")

print("MLP test finished.")
