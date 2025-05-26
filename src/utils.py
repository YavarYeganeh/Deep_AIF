import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import sys
import time
import scipy.stats as st


np_precision = np.float32
buffer_index = 10 + 1  # 11


def reparameterize(x):
    """
    Reparameterization trick when 'x' contains both mean and logvar concatenated along dimension 1.
    The first half of 'x' are means and the second half are log-variances.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape (batch_size, 2*d), where the first d values along dim=1 are means,
        and the next d values are log-variances.

    Returns
    -------
    torch.Tensor
        A sample from N(mean, var), shape (batch_size, d).
    """
    # Determine size
    d = x.size(1) // 2

    # Split into mean and logvar
    mean = x[:, :d]
    logvar = x[:, d:]

    # Sample epsilon from a standard normal distribution
    eps = torch.randn_like(mean)

    # Reparameterization: z = mean + exp(0.5 * logvar) * eps
    return mean + eps * torch.exp(0.5 * logvar)


def prod_planner(o, batch_id, model):

    if model.decision == 'aif':

        o = torch.from_numpy(o).float().view(1,-1)

        Ppi, ref = model.policy(o)

        Ppi = Ppi.view(-1).detach().numpy()

        pi_choice = np.random.choice(7,p=Ppi)

        model.o_i[batch_id].append(o)
        model.o_t[batch_id].append(o)

        model.a_t[batch_id].append(ref)
        model.a_i[batch_id].append(ref)

        return pi_choice
    
    elif model.decision == 'random':

        pi_choice = np.random.choice(7)

        return pi_choice
    
    elif model.decision == 'all-on':
    
        pi_choice = 6  

        return pi_choice    
    
    else:

        raise ValueError(f'The selected decision type, i.e., {model.decision} is not defined!')


def separate_softmax_sigmoid(x: torch.Tensor):

    """
    Applies softmax to the first 11 values (buffer), then to each of the next 6 sets of 5 values (machine states),
    and finally applies sigmoid to the last 3 values (rewards).

    Input structure: 10 + 1 + (6*5) + 3 = 44 total elements per sample.
    - x[:, :11] -> softmax over these 11 values
    - x[:, 11:11+(6*5)] -> reshaped into (batch, 6, 5) and softmax applied over the last dimension
    - x[:, -3:] -> sigmoid

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape (batch_size, 44).

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, 44), where:
          * The first 11 values are a softmax distribution.
          * The next 30 values (6*5) are 6 separate softmax distributions, concatenated.
          * The last 3 values are passed through a sigmoid.
    """

    # Softmax for the first 11 entries
    buffer_softmax = F.softmax(x[:, :buffer_index], dim=-1)

    # Handle the 6 machine categories (6*5=30 values)
    machine_cat = x[:, buffer_index:buffer_index + 6*5]  # shape: (batch, 30)
    machine_cat = machine_cat.view(-1, 6, 5)            # shape: (batch, 6, 5)
    machine_cat_softmax = F.softmax(machine_cat, dim=-1) # softmax along the last dimension
    machine_cat_softmax = machine_cat_softmax.view(-1, 6*5) # reshape back to (batch, 30)

    # Sigmoid for the last 3 entries
    reward_sigmoid = torch.sigmoid(x[:, -3:])

    # Concatenate all parts
    return torch.cat([buffer_softmax, machine_cat_softmax, reward_sigmoid], dim=-1)


# set random seeds for reproducibility
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for CPU
    torch.use_deterministic_algorithms(True)
    # for cuDNN (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # improve unnecessary runtimes if the input is variable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# Simple logger and works well for redirecting all print statements without changing the rest of the code
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.console = sys.stdout
        self.file = open(filename, "w")
    
    def write(self, message):
        self.console.write(message)  # Print to console
        self.file.write(message)    # Write to file
    
    def flush(self):
        self.console.flush()
        self.file.flush()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def record_interaction(self, o0, ref, oh, r ):
        experience = (o0, ref, oh, r )
        self.add(experience)

    def clear_buffer(self):
        self.buffer = []
        self.position = 0

    def get_last_interaction(self):
        if len(self.buffer) < 2:
            return None, None, None, None

        last_experience = self.buffer[self.position - 1]
        second_last_experience = self.buffer[self.position - 2]

        return second_last_experience, last_experience

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position % self.capacity] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


def batch_observe(env, model, sim_interval, batch_size, steps):

    # Step 1: Clear previous observations
    for i in range(batch_size):
        model.o_t[i] = []  # Clear observation lists
        model.a_t[i] = []  # Clear action lists

    # Step 2: Run simulation to gather at least #steps new observations
    s = 0
    sim_time = env.now
    while True:
        env.run(until=sim_time + sim_interval)
        sim_time += sim_interval

        # Check if at least `steps` observations are available for all batches
        if all(len(model.o_t[b]) > steps for b in range(batch_size)):
            break

        s += 1
        if s >= 1000 * steps:  # Timeout check
            for j in range(batch_size):
                print(f"\nThe workstation {j} has {len(model.o_t[j])} observations after {s * sim_interval} seconds. ")
            raise Exception('Simulation ran too long without gathering enough observations for all batches!')

    # Step 3: Extract tensors from the gathered observations
    o0 = [model.o_t[b][0] for b in range(batch_size)]
    o0 = torch.stack(o0).view(batch_size, -1)

    oh = [model.o_t[b][steps] for b in range(batch_size)]
    oh = torch.stack(oh).view(batch_size, -1)

    r = oh[:, -1].reshape(batch_size, 1)  # Extract the reward (last element of `oh`)

    ref = [model.a_t[b][0] for b in range(batch_size)]
    ref = torch.stack(ref).view(batch_size, -1)

    return o0, ref, oh, r 


""" creating/resting dictionary for observations and efe """
def reset_recordings(model, batch_size):

    model.o_t = {}
    model.o_i = {}
    model.a_t = {}
    model.a_i = {}
    model.efe_records = {}
    for i in range(batch_size):
        model.o_t[i] = []
        model.o_i[i] = []
        model.a_t[i] = []
        model.a_i[i] = []
        model.efe_records[i] = {'G': [], 'term0': [], 'term1': [], 'term2': []} # this step is necessary as EFE for the training and training should be recorded separately


def kl_div_gaussian(q_mean, q_logvar, p_mean, p_logvar):
    """
    Compute KL divergence D_KL[Q||P] for two Gaussians:
    Q ~ N(q_mean, exp(q_logvar))
    P ~ N(p_mean, exp(p_logvar))

    Returns element-wise KL: [batch_size, state_dim]
    """
    var_q = torch.exp(q_logvar)
    var_p = torch.exp(p_logvar)

    # KL per dimension
    # KL = 0.5 * [ (var_q / var_p) + ((p_mean - q_mean)^2 / var_p) - 1 + (log(var_p) - log(var_q)) ]
    kl = 0.5 * ( (var_q / var_p)
                + ((p_mean - q_mean)**2 / var_p)
                - 1
                + (p_logvar - q_logvar) )
    return kl


def po_max_sampler(po):

    po_dim = 10 + 1 + 6*5 + 3
    if po.shape[-1] != po_dim:
        raise ValueError("The dimension of po is not 10+1+(6*5)+3, which is required.")

    batch_size = po.size(0)
    buffer_index = 10 + 1  # buffer part dimension

    # Max along the first segment (10+1)
    max_buff = torch.argmax(po[:, :buffer_index], dim=1)
    # One-hot encode this selection
    spo = F.one_hot(max_buff, num_classes=po[:, :buffer_index].shape[1]).float()

    machine_shape = po[:, buffer_index: buffer_index + 5].shape[1]

    # For each of the 6 machine states:
    for i in range(6):
        start_idx = buffer_index + i*5
        end_idx = buffer_index + (i+1)*5
        max_state = torch.argmax(po[:, start_idx:end_idx], dim=1)
        spo = torch.cat([spo, F.one_hot(max_state, num_classes=machine_shape).float()], dim=1)

    # Append the reward part as is:
    # reward shape: [batch_size, 3]
    reward = po[:, -3:].float()
    spo = torch.cat([spo, reward], dim=1)

    return spo


def compute_term1(po):
    """
    Computes an approximation to a log-likelihood-like term for reward predictions with modified continuous penalty.

    This is a heuristic approximation:
    term1 = - E [ log P(o|pi) ] ~ -log(1 - MAE + eps), where MAE = |1 - pred_r| and pred_r = o[:, -1].
    The closer pred_r is to 1, the smaller MAE becomes, making (1 - MAE) close to 1,
    and thus term1 close to 0. As pred_r deviates from 1, MAE increases, 
    reducing (1 - MAE) and increasing term1.

    Note: This is not a true log-likelihood, but a heuristic that encourages pred_r to approach 1.

    Args:
        o (torch.Tensor): A tensor of shape [batch_size, D] where the last element 
                          in each row (o[:, -1]) is the predicted reward.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, 1] containing the approximation 
                      term1 for each sample, scaled by a multiplier.
    """

    # pred_r: predicted reward, shape [batch]
    pred_r = po[:, -1]

    # Compute MAE = |1 - pred_r| as perfect reward is 1
    mae = torch.abs(1.0 - pred_r)

    # Add a small epsilon to avoid log(0)
    eps = 1e-7
    term1 = - torch.log(1.0 - mae + eps)

    # Multiply the final loss by 
    # term1 = term1 * 10
    term1 = term1 * 1 # as mean (1-d) used for the rest of terms

    # Reshape to [batch_size, 1]
    term1 = term1.view(-1, 1)

    return term1


def entropy_normal_from_logvar(logvar):
    # logvar: [batch, D]
    # H_i = 0.5 * (1 + log(2π)) + 0.5 * logvar_i for each element
    entropy = 0.5 * (1.0 + np.log(2 * np.pi) + logvar)
    return entropy  # Shape: [batch_size, D]


def entropy_bernoulli(p):
    eps = 1e-7
    p_clamped = torch.clamp(p, eps, 1 - eps)
    return -(p_clamped * torch.log(p_clamped) + (1 - p_clamped)*torch.log(1 - p_clamped))


class Timer:
    
    def __init__(self):
        """Initialize the Timer class by setting the last time to the current time."""
        self.last_time = time.perf_counter()

    def elapsed(self):
        """
        Calculate and return the time elapsed since the last call to this method.

        Returns:
            float: The elapsed time in seconds since the last call.
        """
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time
        print(f"Elapsed time: {elapsed_time:.5f} seconds")
        self.last_time = current_time

        return elapsed_time
    
    
def state_activation(x, lambda_s=1.5):
    
    """
    Ensures mu is [-1, 1] and var is (0, lambda_s).

    First half -> tanh for mu (range: [-1, 1])
    Second half -> log of scaled sigmoid for log(var),
                   where var is in (0, lambda_s).

    Arguments:
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, 2 * half_size).
    lambda_s : float
        Upper bound scaling factor for variance. Default is 1.5.
    """

    # Determine half of the final dimension
    half_size = x.shape[-1] // 2
    
    # Split the input tensor
    first_half = x[:, :half_size]   # For mean
    second_half = x[:, half_size:]  # For log-variance
    
    # Constrain the mean to [-1, 1]
    first_half_tanh = torch.tanh(first_half)
    
    # Constrain variance to be in (0, lambda_s)
    second_half_sigmoid = torch.sigmoid(second_half)            # (0, 1)
    second_half_sigmoid_scaled = lambda_s * second_half_sigmoid # (0, lambda_s)
    second_half_log = torch.log(second_half_sigmoid_scaled + 1e-10)
    
    # Concatenate (mu, log(var)) along last dimension
    result = torch.cat([first_half_tanh, second_half_log], dim=-1)

    return result


def compute_stats(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)
    
    if n > 1:
        margin_error = st.t.ppf((1 + confidence) / 2, n - 1) * (std / np.sqrt(n))
    else:
        margin_error = 0  # No margin of error for single sample

    return mean, std, margin_error