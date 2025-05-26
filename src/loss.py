
import torch
import torch.nn.functional as F

import src.utils as u


def calculate_loss_transition(qsh_mean, qsh_logvar, psh_mean, psh_logvar):

    """ 
    Loss = kl_div_s = Eqpi D_kl[Q(sh)||P(sh|s0,pi)]
    """

    # Compute KL divergence per dimension: [batch_size, state_dim]
    kl_div_s_anal = u.kl_div_gaussian(qsh_mean, qsh_logvar, psh_mean, psh_logvar)

    # mean over the state dimensions and then mean over batch
    loss = kl_div_s_anal.mean()

    return loss

def calculate_loss_encoder_decoder(beta, oh, poh, qsh_mean, qsh_logvar):

    """ 
    Loss = Term 1 + Term 2
    Term 1 = Eq[log P(o1|s1)] -> Reconstruction loss (oh to be reconstructed from prediction by encoding o0 and transition using the policy reference and then decoding it)
    Term2 = Beta * Eqpi D_kl[Q(s1)||N(0.0,1.0)]
    """

    # Separate the one-hot (binary) part for buffer and machines, and numeric part for rewards
    oh_b = oh[:, 0:11]    # Ground truth for binary features of buffer level
    oh_m = oh[:, 11:-3]    # Ground truth for binary features of machine states
    oh_r = oh[:, -3:]     # Ground truth for numeric features

    po1_b = poh[:, 0:11]  # Predicted probabilities for binary features of buffer level
    po1_m = poh[:, 11:-3]  # Predicted probabilities for binary features of machine states
    po1_r = poh[:, -3:]   # Predicted numeric values

    # Compute BCE loss for the binary part of buffer level
    bce_b = F.binary_cross_entropy(po1_b, oh_b, reduction='mean') 

    # Compute BCE loss for the binary part of machine states
    bce_m = F.binary_cross_entropy(po1_m, oh_m, reduction='mean')

    # Compute an approximation to a log-likelihood-like term for reward predictions with modified continuous penalty
    mae = torch.abs(po1_r - oh_r)
    mse_r = - torch.log(1.0 - mae + 1e-7) # Add a small epsilon to avoid log(0)
    mse_r = mse_r.mean()

    # Combine the three elements for reconstruction loss scaled by significance
    term_1 = (2/7) * bce_b + (1/7) * bce_m + (4/7) * mse_r

    term_2 = beta * u.kl_div_gaussian(qsh_mean, qsh_logvar, torch.zeros_like(qsh_mean), torch.zeros_like(qsh_logvar))
    term_2 = term_2.mean()

    loss = term_1 + term_2

    return loss, mae.mean(), mse_r, bce_b, bce_m
    

def train_transition(model, optimizer, o0, ref, oh, samples): 

    model.train()  # Ensure training mode if needed
    model.zero_grad()  # Zero out gradients of parameters in all modules that are being trained with three optimizers

    # batch_size = o0.size(0)
    device = o0.device

    loss = torch.zeros(1, dtype=model.precision, device=device)

    qs0 = model.encoder(o0)

    qsh = model.encoder(oh)
    state_dim = qsh.size(1) // 2 # split  psh into mean and logvar
    qsh_mean = qsh[:, :state_dim]
    qsh_logvar = qsh[:, state_dim:]

    for i in range(samples):

        qs0_s = u.reparameterize(qs0)
        qs0_s_ref = torch.cat([qs0_s, ref], dim=1)
        
        psh = model.transition(qs0_s_ref)
        psh_mean = psh[:, :state_dim]
        psh_logvar = psh[:, state_dim:]

        loss = loss + calculate_loss_transition(qsh_mean, qsh_logvar, psh_mean, psh_logvar)

    loss = loss/float(samples)  # to prevent large gradients as results of more samples
    
    loss.backward(retain_graph=False)         # Backpropagate to compute gradients
    optimizer.step()         # Update parameters

    return loss.item() 


def train_encoder_decoder(model, optimizer, beta, o0, ref, oh, samples):
    
    model.train()  
    model.zero_grad()  # zero out gradients

    device = o0.device
    
    # Initialize accumulators as tensors
    loss_acc = torch.zeros(1, dtype=model.precision, device=device)
    mae_acc = torch.zeros(1, dtype=model.precision, device=device)
    mse_r_acc = torch.zeros(1, dtype=model.precision, device=device)
    bce_b_acc = torch.zeros(1, dtype=model.precision, device=device)
    bce_m_acc = torch.zeros(1, dtype=model.precision, device=device)

    qs0 = model.encoder(o0)
    state_dim = qs0.size(1) // 2

    qsh = model.encoder(oh)
    qsh_mean = qsh[:, :state_dim]
    qsh_logvar = qsh[:, state_dim:]
    
    # Accumulate loss & metrics across multiple samples
    for _ in range(samples):

        qs0_s = u.reparameterize(qs0)
        qs0_s_ref = torch.cat([qs0_s, ref], dim=1)
        
        psh = model.transition(qs0_s_ref)
        psh_s = u.reparameterize(psh)
        
        poh = model.decoder(psh_s)  
        
        # Compute loss and metrics for current sample
        new_loss, mae_step, mse_r_step, bce_b_step, bce_m_step = calculate_loss_encoder_decoder(
            beta, oh, poh, qsh_mean, qsh_logvar)

        # Accumulate (no in-place ops)
        loss_acc = loss_acc + new_loss
        mae_acc = mae_acc + mae_step
        mse_r_acc = mse_r_acc + mse_r_step
        bce_b_acc = bce_b_acc + bce_b_step
        bce_m_acc = bce_m_acc + bce_m_step
    
    # Average over the number of samples
    loss_final = loss_acc / float(samples)
    mae_final = mae_acc / float(samples)
    mse_r_final = mse_r_acc / float(samples)
    bce_b_final = bce_b_acc / float(samples)
    bce_m_final = bce_m_acc / float(samples)

    # Backpropagation & optimization
    loss_final.backward(retain_graph=False)
    optimizer.step()

    # Return the averaged metrics (converted to Python floats)
    return loss_final.item(), mae_final.item(), mse_r_final.item(), bce_b_final.item(), bce_m_final.item()


def train_generative(model, optimizer_transition, optimizer_encoder_decoder, beta, o0, ref, oh, samples=1):

    """
    train_transition updates part of the model, and then ww need to compute gradients for train_encoder_decoder using a new computational graph based on the model's new state. Accordingly, we separate the updates and compute outputs of model each time. 
    """

    l_transition = train_transition(model, optimizer_transition, o0, ref, oh, samples)

    l_encoder_decoder, mae, mse_r, bce_b, bce_m = train_encoder_decoder(model, optimizer_encoder_decoder, beta, o0, ref, oh, samples)

    return l_transition, l_encoder_decoder, mae, mse_r, bce_b, bce_m


def train_policy(model, optimizer, efe):

    # model.train()  # as efe (the forward pass) is done outside this step is not necessary! 
    model.zero_grad()  # Zero out gradients of parameters in all modules that are being trained with three optimizers

    loss = efe
    loss = loss.mean()     # loss is batch-wise and we want a single scalar
    
    loss.backward(retain_graph=False)         # Backpropagate to compute gradients
    optimizer.step()         # Update parameters
