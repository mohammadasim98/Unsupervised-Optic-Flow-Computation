# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""
import tensorflow as tf
from energy import get_energy
from tqdm import trange

def compute_loss(pred, alpha, epsilon, lmbda, weights, of_gt, train):
    """
    Compute Variational Energy Loss, and EPE
    """
    
    epe = tf.reduce_mean(tf.norm(pred[0][:, :, :, 0:2]-of_gt, axis=-1))
    if train:
        
        # Compute energy for each scale 
        energy_0 = get_energy(pred[0], alpha, epsilon, lmbda)
        energy_1 = get_energy(pred[1], alpha, epsilon, lmbda)
        energy_2 = get_energy(pred[2], alpha, epsilon, lmbda)
        energy_3 = get_energy(pred[3], alpha, epsilon, lmbda)
        energy_4 = get_energy(pred[4], alpha, epsilon, lmbda)
        
        # Weighted summation of the energies of all scales
        energy = weights[0]*energy_0 + weights[1]*energy_1 + weights[2]*energy_2 + weights[3]*energy_3 + weights[4]*energy_4
        return energy, epe
    
    else:
        return 0, epe

@tf.function
def train_step(batch, model, optimiser, alpha, epsilon, lmbda, weights, train=True):
    """ 
    Single Forward and Backpropagation Step
    """
    
    with tf.GradientTape(persistent=False) as tape:
        pred = model(batch[0])
        energy_loss, epe = compute_loss(pred, alpha, epsilon, lmbda, weights, batch[1], train)
    
    if(train):
        grad = tape.gradient(energy_loss, model.trainable_variables)
        optimiser.apply_gradients(zip(grad, model.trainable_variables))
        
    return tf.reduce_mean(energy_loss), epe

def train(data, model, settings):
    """
    Training Loop
    """
    
    # Fetch config from settings
    pbar = trange(settings.epochs)
    optimiser = tf.keras.optimizers.Adam(settings.lr)
    alpha = tf.Variable(settings.alpha, trainable=False)
    epsilon = tf.Variable(settings.epsilon, trainable=False)
    lmbda = tf.Variable(settings.lmbda, trainable=False)
    weights = settings.multi_scale_weights
    patience = settings.patience
    lr_decay = settings.lr_decay
    
    losses = []
    counter = 0
    history = {'train': {'energy': [], 'epe': []}, 'val': {'epe': []}, 'lr': []}
    for i in pbar:
        iter_losses = []
        train_epe = []
        val_epe = []
        history['lr'].append(optimiser.lr.numpy())
        for batch in data[0]:
            loss, epe = train_step(batch, model, optimiser, alpha, epsilon, lmbda, weights, train=True)
            iter_losses.append(loss)
            train_epe.append(epe)
            
        epoch_loss = tf.reduce_mean(iter_losses).numpy()
        train_epoch_epe = tf.reduce_mean(train_epe).numpy()
        history['train']['energy'].append(epoch_loss)
        history['train']['epe'].append(train_epoch_epe)
        if(min(history['train']['energy']) < epoch_loss):
            counter += 1
            
        else:
            counter = 0
            model.save('ckpt.h5')
            
        if(patience is not None and counter == patience):
            counter = 0
            optimiser.lr.assign(optimiser.lr.numpy()*lr_decay)
            print(f"Learning rate decayed to: {optimiser.lr.numpy()}, Minimum (energy, epe) was: ({min(history['train']['energy'])}, {min(history['train']['epe'])})")
       
        val_losses = []
        if(data[1] is not None):
            for batch in data[1]:
                loss, epe = train_step(batch, model, optimiser, alpha, epsilon, lmbda, weights, train=False)
                val_epe.append(epe)
                
        val_epoch_epe = tf.reduce_mean(val_epe).numpy()
        history['val']['epe'].append(val_epoch_epe)
        pbar.set_description(f"energy:  {epoch_loss}, train_epe: {train_epoch_epe}, val_epe: {val_epoch_epe}, patience_count: {counter}")
        
    return model, history