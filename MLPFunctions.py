import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time

DEBUG = False


class TF:
    # defines types of transferfunctions which can be included into single neuron
    def __init__(self, tfType):
        self.tfType = tfType

    def calcOut(self, activation):
        # depending on type of transferfunction, activation is calculated
        # with bias, neuron output is calculated
        if( self.tfType is "tfStep" ):
            out = []
            for act in activation:
                if( act > 0 ):
                    out.append(1)
                else:
                    out.append(0)

        elif( self.tfType is "nothing"):
            out = activation

        elif( self.tfType is "tanh" ):
            out = np.empty(0)
            for act in activation:
                res = 2 * ( (1) / (1 + np.exp(-2*act)) ) - 1    # use logsig to calc tanh with less exp()
                out = np.append(out, res)

        elif( self.tfType is "logsig" ):
            out = np.empty(0)
            for act in activation:
                res = (1) / (1 + np.exp(-act)) 
                out = np.append(out, res)

        elif( self.tfType is "relu" ):
            out = np.empty(0)
            for act in activation:
                if( act > 0 ):
                    res = act
                else:
                    res = 0
                out = np.append(out, res)

        else:
            print("ERROR: Unknown TF")
            return -1

        return out


    def d2dx(self,inputVec):
        # defines the derivative of the tf
        if( self.tfType is "tfStep" ):
            print("ERROR: Unknown TF")
            return -1

        elif( self.tfType is "nothing"):
            out = np.ones(len(inputVec))

        elif( self.tfType is "tanh" ):
            out = np.empty(0)
            for x in inputVec:
                out = np.append(out, self.d2dx_tanh(x))

        elif( self.tfType is "logsig" ):
            out = np.empty(0)
            for x in inputVec:
                out = np.append(out, self.d2dx_logsig(x))

        elif( self.tfType is "relu" ):
            out = np.empty(0)
            for x in inputVec:
                out = np.append(out, self.d2dx_relu(x))

        else:
            print("ERROR: Unknown TF")
            return -1
        return out

    def d2dx_tanh(self, x):
        tanh = 2 * ( (1) / (1 + np.exp(-2 * x)) ) - 1    # use logsig to calc tanh with less exp()
        #print(tanh)
        return (1 - tanh**2)

    def d2dx_logsig(self,x):
        logsig = (1) / (1 + np.exp(-x)) 
        return (logsig * (1 - logsig))

    def d2dx_relu(self,x):
        return 1 if x > 0 else 0

class LAYER:
    def __init__(self, neuInLay, layerNr, tfInLay, inputVec):
        self.tfType = tfInLay
        self.tf = TF(self.tfType)
        self.layerNr = layerNr
        self.neuInLay = neuInLay

        self.inputVec = []  #inputVec.extend([1])    # extend with bias
        self.outputVec = []
        self.weights = 2 * ((np.random.rand(neuInLay, len(inputVec))) -0.5)  #* layerNr  
#np.ones([neuInLay, len(inputVec)])
        self.activation = []

        self.dact2dw = np.zeros([neuInLay, len(inputVec)])
        self.dy2dact = np.zeros([neuInLay, 1])
        self.dE2dy = np.zeros([neuInLay, 1])
        self.deltaMat = np.zeros([neuInLay, neuInLay])
        self.deltaVec = np.zeros([neuInLay, 1])
        self.deltaWeights = np.zeros(self.weights.shape)

    def calcOut(self):
        """
        > calculates the output of neuron
        therefore the neuron activation is scaled to the number of inputs. 
        Reason: out = tf(act) with act = sum(weights x inputVec)
        for high input values, out goes into saturation.  
        to use this scale, input and teacher have to be scaled too!!
        """
        self.activation = np.matmul(self.weights, self.inputVec) # calculates activation 
        scaleAct = 1#np.amax(abs(self.activation))
        self.activation = self.activation / scaleAct
        self.outputVec = self.tf.calcOut(self.activation)


class MLP:
    def __init__(self, inputVec):
        self.numNeuLay = []     # number of neurons per layer
        self.tfTypeLay = []      # typ of transferfunc in this layer
        self.layer = []         # contains one array per layer, which contains all neurons and their weights
        self.inputVec = inputVec   # input vector of first layer
        self.outputVec = []        # output vector of last layer
        self.outputLay = []     # vector of outputvectors of each layer
        self.alpha = 0.1          # learn rate
        self.teacherVec = []        # teacher vector scaled/unscaled

    def add(self, neuInLay, layerNr, tfInLay):
        """
        takes desired number of neurons and type of tf for this layer as input
        creates new neuron and appends it to layer list. 
        layer list is appended to global layer list
        """
        self.numNeuLay.append(neuInLay)
        self.tfTypeLay.append(tfInLay)

        """
        > if outputLay vector is empty, the first layer is going to be added
        first layer has no active weigths, gets filled with identity matrix
        all other layers inputVec is extended with bias [1]
        """
        if not self.outputLay: 
            self.layer.append(LAYER(neuInLay, layerNr, tfInLay, self.inputVec))
            self.layer[0].weights = np.eye(len(self.inputVec))     # first layer: no calculation; no weights

        else:
            inputVec = np.append( self.outputLay[-1], [1] )  # output from last layer is input to this layer
            self.layer.append(LAYER(neuInLay, layerNr, tfInLay, inputVec) )  
        self.outputLay.append(np.zeros(neuInLay))

    def feedForward(self):
        """
       goes through all layer of mlp
       chooses inputVec depending on which layer is going to be computed
       e.g. first layer takes global inputVec, layer n takes output n-1 as inputVec
       """
        for layer, layerNr in zip(self.layer, range(0, len(self.layer))):
            if( 0 == layerNr ):
                layer.inputVec = self.inputVec
            else:
                layer.inputVec = np.append( self.outputLay[layerNr - 1], [1] )
            # > calls calculation of output
            layer.calcOut()
            self.outputLay[layerNr] = layer.outputVec
        self.outputVec = self.outputLay[-1]    # set last element of outputLay to output

    def backprop(self, teacherVec):
        """
        adapts the weights of the output and hidden neurons of the MLP
        using the Backpropagation algorithm

        when a Matrix is uesed:
            rows represent the neuron in this layer
            columns represent the inputs into this neuron 
        """

        outputLayer = True
        self.teacherVec = teacherVec

        """
        1.) go through all the layer 
        start with the last, stop before the first
        """
        for layer, layerNr in zip(reversed(self.layer), reversed(range(0,len(self.layer))) ):
            if(DEBUG):
                print("\n")
                print("\nLayerNr %d: " % layerNr)
                print("\nweights = ", end="")
                print(layer.weights)
            # > no weight manipulation in the first layer; leave loop
            if( layerNr is 0 ):
                if(DEBUG):
                    print("\nInputLayer BREAK")
                break   

            """
            2.) fill Mat dact2dw with values
            dact2dw holds the output values of the previous layer 
            N_l x N_(l-1)+1
            
            row represents the neuron in current layer
            each neuron gets the same input from the previous layer (including bias); stored in columns
            """
            for neuNr in range(0, layer.neuInLay):
                layer.dact2dw[neuNr,:] = np.append( self.outputLay[layerNr - 1], [1] )  # append 1 for bias

                if(DEBUG):
                    print("Input in NeuronNr %d: " %neuNr )
                    print(np.append( self.outputLay[layerNr - 1], [1] ))
            if(DEBUG):
                print("\ndact2dw = ", end="")
                print(layer.dact2dw)

            """
            3.) fill Mat dy2dact with values
            holds the values of dtf/dt 
            N_l x 1
            """
            layer.dy2dact = (layer.tf.d2dx(layer.activation)).reshape(layer.neuInLay,1)
            if(DEBUG):
                print("\nactivation of Layer = ", end="")
                print(layer.activation)
                print("\ndy2dact = ", end="")
                print(layer.dy2dact)

            """
            4.) fill dE2dy with values
            in output layer dE2dy equals output - teacher
            N_l x 1
            For hidden layer different calculation
            """
            if( outputLayer is True):
                layer.dE2dy = (self.outputVec - self.teacherVec).reshape(layer.neuInLay, 1)
                
                """
                4.1) fill deltaMat, deltaVec
                deltaVec equals:
                delta = dE/dy * dy/dact   (elementwise)
                use as matrix for direct multiplication with dact/dw
                so dw can be computet directly
                """
                layer.deltaVec = np.multiply(layer.dE2dy, layer.dy2dact)
                layer.deltaMat = np.diag(layer.deltaVec.ravel())
                if(DEBUG):
                    print("\nteacherVec = ", end="")
                    print(self.teacherVec)
                    print("\noutputVec = ", end="")
                    print(self.outputVec)
                    print("\ndE2dy = ", end="")
                    print(layer.dE2dy)
                    print("\ndeltaVec = ", end="")
                    print(layer.deltaVec)
                    print("\ndeltaMat = ", end="")
                    print(layer.deltaMat)
            else:
                """
                for all hidden layer
                delta_l depends on delta_(l+1) and weights_(l+1)
                
                dE/dy has as many elements as the current layer has outputs (bias not included)
                
                dE/dy:
                dE/dy1_l = ( delta[1]_(l+1) * w(1,1)_(l+1) ) + ( delta[2]_(l+1) * w(2,1)_(l+1) )
                equals:
                dE/dy1_l = delta_(l+1).T o w(:,1)_(l+1)
                deltaVec.T der nachfolgenden Schicht, skalar mit den Spalten der Wichtungsmatrix (neurone der nachfolgenden Schicht) 

                the equation in literature (Brauer skript) request a negaation of the dE/dy1_l = delta_(l+1).T o w(:,1)_(l+1)
                BUT:! here, deltaVec is :
                layer.deltaVec = np.multiply(layer.dE2dy, layer.dy2dact)

                compared to skript this equals (-delta)

                so no negation needed during computation of dE/dy
                """
                if(DEBUG):
                    print("\ndE2dy:")

                for neuNr in range(0, layer.neuInLay):
                    # for delta_l iterate through delta_l+1 and weights_l+1
                    layer.dE2dy[neuNr] = (np.dot( self.layer[layerNr+1].deltaVec.T, self.layer[layerNr+1].weights[:,neuNr] ))
                    if(DEBUG):
                        print("\n\tCalculate NeuronNr %d:" % neuNr)
                        print("\tdeltaVec layerNr: %d = " % (layerNr+1) )
                        print("\t",self.layer[layerNr+1].deltaVec)
                        print("\tTrue weights layerNr: %d for this Neurons output = " % (layerNr+1) )
                        print("\t",self.layer[layerNr+1].weights[:,neuNr])
                        print("\tdE2dy[%d] = " % neuNr)
                        print("\t", layer.dE2dy[neuNr])

                """
                4.1) fill deltaMat, deltaVec
                for direct computation of dw, generate deltaMat
                """
                layer.deltaVec = np.multiply(layer.dE2dy, layer.dy2dact)
                layer.deltaMat = np.diag(layer.deltaVec.ravel())
                if(DEBUG):
                    print("\ndE2dy = ", end="")
                    print(layer.dE2dy)
                    print("\ndeltaVec = ", end="")
                    print(layer.deltaVec)
                    print("\ndeltaMat = ", end="")
                    print(layer.deltaMat)

            outputLayer = False

        if(DEBUG):
            print("\nCalculate new Weights")
            print("\nlearnRate alpha = %f" %self.alpha)

        """
        6.) go again through all layer and calculate the new weights
        """
        for layer, layerNr in zip(reversed(self.layer), reversed(range(0,len(self.layer))) ):
            if(DEBUG):
                print("\n\tLayerNr %d:" % layerNr)
            # > no weight manipulation in the first layer; leave loop
            if( layerNr is 0 ):
                if(DEBUG):
                    print("\n\tinputLayer BREAK")
                break   

           # 7.) calculate Delta weights
            layer.deltaWeights = -(self.alpha) * np.matmul(layer.deltaMat, layer.dact2dw)
            layer.weights += layer.deltaWeights
            if(DEBUG):
                print("\n\tdeltaWeights = ", end="")
                print("\t", layer.deltaWeights)
