# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""
import tensorflow as tf

class Model():
    def SobelXInitialiser(self, depth):
        def SobelXInitialiser(shape, dtype=tf.float32):
            """
            Sobel-X Kernel
            """
            
            sobel_x = tf.constant(
                [
                    [1, 0, -1], 
                    [2, 0, -2], 
                    [1, 0, -1], 
                ], dtype=dtype )
            sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
            sobel_x = tf.tile(sobel_x, (1, 1, depth, 1))
            return sobel_x

        return SobelXInitialiser

    def SobelYInitialiser(self, depth):
        def SobelYInitialiser(shape, dtype=tf.float32):
            """
            Sobel-Y Kernel
            """
            
            sobel_y = tf.constant(
                [
                    [1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1], 
                ], dtype=dtype )
            sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))
            sobel_y = tf.tile(sobel_y, (1, 1, depth, 1))
            return sobel_y
        
        return SobelYInitialiser

    def encoder_blk(self, inp: tf.Tensor, layers: int=2, filters: int=32, 
                    pool: bool=True, activation=None, residual=True):
        """
        Encoder Block
        """
        
        node_inp = tf.keras.layers.Conv2D(filters, (1, 1), 
                                          activation=activation, 
                                          padding='SAME')(inp)
        node = tf.keras.layers.Conv2D(filters, (3, 3), 
                                      activation=activation,
                                      padding='SAME')(node_inp)
        for i in range(layers-1):
            node = tf.keras.layers.Conv2D(filters, (3, 3), 
                                          activation=activation, 
                                          padding='SAME')(node)
            
        if residual:
            node = tf.keras.layers.Add()([node, node_inp])
            
        if pool:
            pool = tf.keras.layers.AveragePooling2D()(node)
            
        return node, pool

    def decoder_blk(self, inp: tf.Tensor, node: tf.Tensor, layers: int=2, filters: int=32, 
                    pool: bool=True, activation=None, residual=True):
        """
        Decoder Block
        """
        
        conv =  tf.keras.layers.Conv2DTranspose(filters, (3, 3), activation=activation, padding='SAME')(inp)
        for i in range(layers-1):
            conv = tf.keras.layers.Conv2DTranspose(filters, (3, 3), activation=activation, padding='SAME')(conv)
            
        ups = tf.keras.layers.UpSampling2D(interpolation='bilinear')(conv)

        # Optic Flow at current scale
        of = tf.keras.layers.Lambda(lambda x : x[:,:,:, 0:2])(ups)
        concat = tf.keras.layers.concatenate([ups, node], axis=-1)
        
        return concat, of
    
    def flow_net(self, inp, activation=None):
        """
        FlowNetSimple-UNet-like Architecture with Residuals Connections.
        """
        
        conv0 = tf.keras.layers.Conv2D(8, (5, 5), activation=activation, padding='SAME')(inp)
        
        # Encoder Blocks
        conv1, pool0 = self.encoder_blk(conv0, layers=2, filters=8, activation=activation, residual=True)
        conv2, pool1 = self.encoder_blk(pool0, layers=2, filters=16, activation=activation, residual=True)
        conv3, pool2 = self.encoder_blk(pool1, layers=2, filters=32, activation=activation, residual=True)
        conv4, pool3 = self.encoder_blk(pool2, layers=2, filters=32, activation=activation, residual=True)
        
        # Pipe
        conv = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='SAME')(pool3)
        conv = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='SAME')(conv)
        of_4 = tf.keras.layers.Lambda(lambda x : x[:,:,:, 0:2])(conv)
        
        # Decoder Blocks
        concat, of_3 = self.decoder_blk(conv, node=conv4, layers=2, filters=32, activation=activation)
        concat, of_2 = self.decoder_blk(concat, node=conv3, layers=2, filters=32, activation=activation)
        concat, of_1 = self.decoder_blk(concat, node=conv2, layers=2, filters=16, activation=activation)
        concat, _ = self.decoder_blk(concat, node=conv1, layers=2, filters=8, activation=activation)
        
        # Garnish
        deconv3 =  tf.keras.layers.Conv2D(8, (3, 3), activation=activation, padding='SAME')(concat)
        deconv3 =  tf.keras.layers.Conv2D(4, (3, 3), activation=activation, padding='SAME')(deconv3)
        of_0 =  tf.keras.layers.Conv2D(2, (3, 3), activation=activation, padding='SAME')(deconv3)

        return of_0, of_1, of_2, of_3, of_4
    
    def downsample(self, inp: tf.Tensor, factor: int=1):
        """
        Downsample Tensors by factor of two using average pooling,
        Designed to downsample the input images for multi-scale coarse-to-fine warping.
        Inputs:
            Shape: (BATCH_SIZE, X, Y, C)
        Outputs:
            Shape: (BATCH_SIZE, X/(2^factor), Y/(2^factor), C)
        """
        
        pool = tf.keras.layers.AveragePooling2D()(inp)
        for _ in range(factor-1):
            pool = tf.keras.layers.AveragePooling2D()(pool)
        return pool
    
    def diff(self, inp: tf.Tensor, input_depth: int=2):
        """
        Differentiator: Designed to perform x- and y- directional differentiation after applying 3x3 gaussian smoothings
        Inputs:
            Shape: (BATCH_SIZE, X, Y, C)
        Outputs:
            Depthwise concatenation in the order [dx, dy] where dx, and dy both have same channels as the inputs
            Shape: [(BATCH_SIZE, X, Y, C), (BATCH_SIZE, X, Y, C)] = (BATCH_SIZE, X, Y, 2*C)
        """
        
        # x-derivative
        xdiff = tf.keras.layers.DepthwiseConv2D((3, 3), (1, 1), depthwise_initializer=self.SobelXInitialiser(input_depth), 
                                                use_bias=None, trainable=False, padding='VALID')(inp)
        xdiff = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(xdiff)
        
        # y-derivative
        ydiff = tf.keras.layers.DepthwiseConv2D((3, 3), (1, 1), depthwise_initializer=self.SobelYInitialiser(input_depth), 
                                                use_bias=None, trainable=False, padding='VALID')(inp)
        ydiff = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(ydiff)
        
        return tf.keras.layers.concatenate([xdiff, ydiff], axis=-1)
    
    def build_network(self, activation, load=False, path=None):
        """
        Defining the full network with differentiation and downsampling blocks
        Nomenclature:
            Inputs ---> Contains previous_frame, current_frame at index 0, 1 respectively
            of_x ---> Optic flow at scale x with u, v at index 0, 1 respectively
            
            Scale-4 ---> Resolution at 8x16
            Scale-3 ---> Resolution at 16x32
            Scale-2 ---> Resolution at 32x64
            Scale-1 ---> Resolution at 64x128
            Scale-0 ---> Resolution at 128x256 (Input Resolution)
                        
            Current frame ---> Frame at instant i
            Previous frame ---> Frame at instant i-1
            
            flow_x ---> Contains optic flow, input frames, and the x, y derivatives at scale x
        """
        
        # Define Input
        inp = tf.keras.Input((128, 256, 2))
        
        # Split input tensor into two separate tensors
        current_frame = tf.keras.layers.Lambda(lambda x : x[:,:,:,1])(inp)
        current_frame = tf.keras.layers.Reshape((128, 256, 1))(current_frame)
        previous_frame = tf.keras.layers.Lambda(lambda x : x[:,:,:,0])(inp)
        previous_frame = tf.keras.layers.Reshape((128, 256, 1))(previous_frame)
        
        # Define Main Network
        of_0, of_1, of_2, of_3, of_4 = self.flow_net(inp, activation=activation)

        # Add 3-stage downsample 
        inp_2x = self.downsample(inp, 1)
        inp_4x = self.downsample(inp, 2)
        inp_8x = self.downsample(inp, 3)
        inp_16x = self.downsample(inp, 4)
        
        # Concatentate the computed Optic Flow and downsampled image at scale  depthwise
        # Perform x- and y- differentiation on the concatenated result
        concat = tf.keras.layers.concatenate([of_0, inp])
        diff = self.diff(concat, input_depth=4)
        flow_0 = tf.keras.layers.concatenate([concat, diff])
        
        concat = tf.keras.layers.concatenate([of_1, inp_2x])
        diff = self.diff(concat, input_depth=4)
        flow_1 = tf.keras.layers.concatenate([concat, diff])

        concat = tf.keras.layers.concatenate([of_2, inp_4x])
        diff = self.diff(concat, input_depth=4)
        flow_2 = tf.keras.layers.concatenate([concat, diff])

        concat = tf.keras.layers.concatenate([of_3, inp_8x])
        diff = self.diff(concat, input_depth=4)
        flow_3 = tf.keras.layers.concatenate([concat, diff])
        
        concat = tf.keras.layers.concatenate([of_4, inp_16x])
        diff = self.diff(concat, input_depth=4)
        flow_4 = tf.keras.layers.concatenate([concat, diff])

        model = tf.keras.Model(inputs=inp, outputs=[flow_0, flow_1, flow_2, flow_3, flow_4])

        if load:
            model.load_weights(path)

        return model
