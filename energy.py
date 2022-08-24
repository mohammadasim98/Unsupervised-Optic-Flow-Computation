# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""

import tensorflow as tf
import tensorflow_addons as tfa

def charbonier_penalizer(x: tf.Tensor, epsilon: float=10e-3):
    """
    Charbonnier Penaliser (Bruhn and Weickert 2005)
    """
    
    return tf.sqrt(x+tf.square(epsilon))

def perona_malik_diffusivity(x, lmbda=1e-2):
    """
    Diffusivity for the Smoothness Term
    Perona-Malik Diffusivity 
    """
    
    return 1/(1 + x/tf.square(lmbda))

def IIDR(frame_diff, ux, uy, vx, vy, lmbda=1e-2):
    """
    Isotropic Image-Driven Regularisation (Alvrez 1999) 
    """
    
    grad_mag = tf.square(ux) + tf.square(uy) + tf.square(vx) + tf.square(vy)
    return perona_malik_diffusivity(frame_diff, lmbda)*grad_mag

def get_energy(flow, alpha, epsilon, lmbda):
    """
    Calulate Variational Energy Loss
    
    Structure of flow (channel/depth wise):
        {u, v, I0, I1, ux, vx, I0x, I1x, uy, vy, I0y, I1y}
    """
    
    # Get the optic flow (u, v) and the corresponding derivatives
    of = flow[:, :, :, 0:2]
    ux = flow[:, :, :, 4]
    vx = flow[:, :, :, 5]
    uy = flow[:, :, :, 8]
    vy = flow[:, :, :, 9]
    
    # Compute x-, y- derivatives of the previous frame
    frame0_diffx = tf.expand_dims(flow[:, :, :, 6], axis=-1)
    frame0_diffy = tf.expand_dims(flow[:, :, :, 10], axis=-1)
    
    # Compute the gradient magnitude for the previous frame
    frame_diff = tf.square(frame0_diffx) + tf.square(frame0_diffy)
    frame_diff = tf.squeeze(frame_diff, axis=-1)
    
    # Get previous and current frame with the corresponding derivatives 
    prev_frame = tf.expand_dims(flow[:, :, :, 2], axis=-1)
    current_frame = tf.expand_dims(flow[:, :, :, 3], axis=-1)
    
    # Warp the previous frame to match the next frame (Make sure to switch the u,v  to v,u)
    # Inverse Bilinear Warping of the current frame using computed optic flow
    warped_frame = tfa.image.dense_image_warp(current_frame, -of[:, :, :, ::-1])

    # Compute the squared differences of the current frame and the warped version of the previous frame
    brightness_loss = tf.square(prev_frame-warped_frame)

    # Compute data term for all pixels using the computed squared differences and the charbonier penalizer
    data_terms = charbonier_penalizer(brightness_loss, epsilon)
    
    # Compute smoothness term using the x-y derivatives of u and v component of optic flow and perona malik diffusivity
    smoothness_term = IIDR(frame_diff, ux, uy, vx, vy, lmbda)
    
    # Sum up results for all pixels
    data_loss = tf.reduce_sum(data_terms, axis=[-3, -2, -1])
    smooth_loss = alpha*tf.reduce_sum(smoothness_term, axis=[-2, -1])

    return data_loss + smooth_loss