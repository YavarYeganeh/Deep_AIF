

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import * 


class Policy(nn.Module):

    def __init__(self, o_dim=1+1+6*5+3, hidden_dims=[64, 32], action_dim=7, dropout_prob=0.05):

        """
        Generates policy probabilities given the sampled observation as well as a signal, which is a reference to the policy for transition (approximation of the policy during prediction and paving the way for gradient during policy optimization).

        For the reference: x has more info therefore better for the reference w.r.t the policy (parameters) 

        Parameters:
        -----------
        o_dim : int
            Number of input features (observation).
        hidden_dims : list of int
            Sizes of the hidden layers.
        action_dim : int
            Number of actions for the policy.
        dropout_prob : float
            Probability of dropout applied after each hidden layer.
        """
        
        super().__init__()
        
        layers = []
        current_dim = o_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            current_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, action_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier (Glorot) initialization for hidden layers with ReLU gain
        relu_gain = nn.init.calculate_gain('relu')
        # For the output layer with a linear activation, the gain is 1.0
        linear_gain = nn.init.calculate_gain('linear')  # This is 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Check if this is the output layer
                if m is self.output_layer:
                    # Output layer (linear gain)
                    nn.init.xavier_uniform_(m.weight, gain=linear_gain)
                else:
                    # Hidden layers (ReLU gain)
                    nn.init.xavier_uniform_(m.weight, gain=relu_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x0):
        h_x = self.hidden(x0)
        x = self.output_layer(h_x)
        pi = F.softmax(x, dim=1) # create probabilities for the planner
        ref = torch.cat([x0, h_x, x], dim=1) # create a reference to the policy for transition
        return pi, ref
 

class Encoder(nn.Module):

    def __init__(self, o_dim=1+1+6*5+3, hidden_dims=[64, 64], state_dim=64, dropout_prob=0.05, lambda_s=1.5):
    
        """
        Encode a sampled observation to a (Gaussian) state distribution in the latent space.
        
        Parameters:
        -----------
        o_dim : int
            Number of input features (observation).
        hidden_dims : list of int
            Sizes of the hidden layers.
        state_dim : int
            Size of the latent representation produced by the network.
        dropout_prob : float
            Probability of dropout applied after each hidden layer.
        """
        super().__init__()
        
        layers = []
        current_dim = o_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            current_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, state_dim + state_dim) # output dimension is twice the state dim as it generates mean and logvar of the Gaussian distribution of state_dim

        self.lambda_s = lambda_s
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier (Glorot) initialization for hidden layers with ReLU gain
        relu_gain = nn.init.calculate_gain('relu')
        # For the output layer with a linear activation, the gain is 1.0
        linear_gain = nn.init.calculate_gain('linear')  # This is 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Check if this is the output layer
                if m is self.output_layer:
                    # Output layer (linear gain)
                    nn.init.xavier_uniform_(m.weight, gain=linear_gain)
                else:
                    # Hidden layers (ReLU gain)
                    nn.init.xavier_uniform_(m.weight, gain=relu_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x) # No softmax here, just return the latent vector (means and logvars of a Gaussian distribution)
        x = state_activation(x, lambda_s=self.lambda_s)
        return x  
    

class Transition(nn.Module):
    
    def __init__(self, state_dim=64, pi_dim=44+32+7, hidden_dims=[64, 64], dropout_prob=0.05, lambda_s=1.5):
    
        """
        Generates a (Gaussian) state distribution using a sampled state in the latent space as well as a signal, which is a reference to the policy (approximation of the policy during prediction and paving the way for gradient during policy optimization). The generated state distribution pertains to a horizon h.
        
        Parameters:
        -----------
        state_dim : int
            Size of the latent representation produced by and fed into the network.
        pi_dim : int
           Size of the reference to the policy. 
        hidden_dims : list of int
            Sizes of the hidden layers.
        dropout_prob : float
            Probability of dropout applied after each hidden layer.
        """
        super().__init__()
        
        layers = []

        current_dim = state_dim + pi_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            current_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, state_dim + state_dim) # output dimension is twice the state dim as it generates mean and logvar of the Gaussian distribution of state_dim

        self.lambda_s = lambda_s
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier (Glorot) initialization for hidden layers with ReLU gain
        relu_gain = nn.init.calculate_gain('relu')
        # For the output layer with a linear activation, the gain is 1.0
        linear_gain = nn.init.calculate_gain('linear')  # This is 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Check if this is the output layer
                if m is self.output_layer:
                    # Output layer (linear gain)
                    nn.init.xavier_uniform_(m.weight, gain=linear_gain)
                else:
                    # Hidden layers (ReLU gain)
                    nn.init.xavier_uniform_(m.weight, gain=relu_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x) # No softmax here, just return the latent vector (means and logvars of a Gaussian distribution)
        x = state_activation(x, lambda_s=self.lambda_s)
        return x  
    

class Decoder(nn.Module):

    def __init__(self, state_dim=64, hidden_dims=[64, 64], po_dim=1+1+6*5+3, dropout_prob=0.05):
    
        """
        Decode a sampled state in the latent space into a specific distribution with observation structure as prediction.
        
        Parameters:
        -----------
        state_dim : int
            Size of the latent representation fed into the network.
        hidden_dims : list of int
            Sizes of the hidden layers.
        po_dim : int
            Number of output features (observation).
        dropout_prob : float
            Probability of dropout applied after each hidden layer.
        """
        super().__init__()
        
        layers = []

        current_dim = state_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            current_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, po_dim) 
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier (Glorot) initialization for hidden layers with ReLU gain
        relu_gain = nn.init.calculate_gain('relu')
        # For the output layer with a linear activation, the gain is 1.0
        linear_gain = nn.init.calculate_gain('linear')  # This is 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Check if this is the output layer
                if m is self.output_layer:
                    # Output layer (linear gain)
                    nn.init.xavier_uniform_(m.weight, gain=linear_gain)
                else:
                    # Hidden layers (ReLU gain)
                    nn.init.xavier_uniform_(m.weight, gain=relu_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        x = separate_softmax_sigmoid(x) # specific distribution with observation structure for prediction.
        return x  
    

class DAIF(nn.Module):
    
    def __init__(self, o_dim=1+1+6*5+3, latent_dims=64, action_dim=7, dropout_prob=0.05, samples=10, calc_mean=False, lambda_s=1.5):

        super().__init__()

        self.policy = Policy(o_dim=o_dim, action_dim=7, dropout_prob=dropout_prob)
        self.encoder = Encoder(o_dim=o_dim, state_dim=latent_dims, dropout_prob=dropout_prob, lambda_s=lambda_s)
        self.transition = Transition(state_dim=latent_dims, pi_dim=o_dim+32+action_dim, dropout_prob=dropout_prob, lambda_s=lambda_s)
        self.decoder = Decoder(state_dim=latent_dims, po_dim=o_dim, dropout_prob=dropout_prob)

        self.decision = 'random'  # this will impact prod_planner in util to initialize with random then it will be changed to self.decision = 'aif' to engage the aif model

        self.precision = torch.float32 

    def freeze_policy_train_generative(self):
        """
        Freeze the parameters of the `policy` module and make all generative modules trainable.
        """
        # Freeze policy parameters
        for param in self.policy.parameters():
            param.requires_grad = False

        # Make generative module parameters trainable
        for module in [self.encoder, self.transition, self.decoder]:
            for param in module.parameters():
                param.requires_grad = True

    def train_policy_freeze_generative(self):
        """
        Make the parameters of the `policy` module trainable and freeze all generative modules.
        """
        # Make policy parameters trainable
        for param in self.policy.parameters():
            param.requires_grad = True

        # Freeze generative modules parameters
        for module in [self.encoder, self.transition, self.decoder]:
            for param in module.parameters():
                param.requires_grad = False


    def forward(self, o):

        qs0 = self.encoder(o)
        pi, ref = self.policy(o)

        qs0_s = reparameterize(qs0)

        qs0_s_ref = torch.cat([qs0_s, ref], dim=1)
        psh = self.transition(qs0_s_ref)

        psh_s = reparameterize(psh)

        poh = self.decoder(psh_s)

        return poh, pi, qs0, qs0_s, psh, psh_s  


    def efe(self, o, calc_mean=False, samples=10):

        """
            Estimate Expected Free Energy using Monte Carlo sampling but lighter compared to the full MC sampling of DAIMC.       
        """

        batch_size = o.size(0)
        device = o.device

        sum_term1 = torch.zeros(batch_size,1, dtype=self.precision, device=device)
        sum_term2 = torch.zeros(batch_size,1, dtype=self.precision, device=device)
        sum_term3 = torch.zeros(batch_size,1, dtype=self.precision, device=device)

        for i in range(samples):
    
            poh, pi, qs0, qs0_s, psh, psh_s = self.forward(o)

            state_dim = qs0.size(1) // 2
            psh_logvar = psh[:, state_dim:]

            poh_s = po_max_sampler(poh) # po is a distribution with different structure with o and can't be directly fed into the encoder

            qsh = self.encoder(poh_s)
            qsh_logvar = qsh[:, state_dim:]

            # rerm1 : - E [ log P(o|pi) ]
            term1 = compute_term1(poh)

            # trerm2 : E [ log Q(s|pi) - log Q(s|o,pi) ] or + E_{Q(θ|π)} [ E_{Q(o_τ|θ, π)}[ H(s_τ|o_τ, π) ] - H(s_τ|π) ]
            H_s_pi = entropy_normal_from_logvar(psh_logvar)
            H_s_o_pi = entropy_normal_from_logvar(qsh_logvar)
            term2 = torch.sum(H_s_o_pi - H_s_pi, dim=1, keepdim=True) 
            # term2 = - torch.sum(H_s_o_pi - H_s_pi, dim=1, keepdim=True) # added the negative as it produced negative outputs ... 
            term2 = torch.mean(H_s_o_pi - H_s_pi, dim=1, keepdim=True) # for scaling as for term 1 one 1-d element used
 
            # term3 : E [ H(o|s,th,pi) - E [ H(o|s,pi) ]
            # term 3.1: Sampling a different θ by a sampling a new qs0_s and then transition and decoding or effectively calling th forward once again
            poh_temp1, pi, qs0, qs0_s_temp1, psh_temp1, psh_s_temp1 = self.forward(o)
            term3_1 = entropy_bernoulli(poh_temp1)

            # Term S.2: Sampling a different psh_s with the same theta psh, i.e. just the reparametrization trick!
            psh_s_temp2 = reparameterize(psh)
            poh_temp2 = self.decoder(psh_s_temp2)
            term3_2 = entropy_bernoulli(poh_temp2)

            # term3 = torch.sum(term3_1 - term3_2, dim=1, keepdim=True)
            term3 = torch.mean(term3_1 - term3_2, dim=1, keepdim=True) # for scaling as for term 1 one 1-d element used
    
            sum_term1 = sum_term1 + term1
            sum_term2 = sum_term2 +  term2
            sum_term3 = sum_term3 + term3
        
        sum_term1 = sum_term1/float(samples)
        sum_term2 = sum_term2/float(samples)
        sum_term3 = sum_term3/float(samples)

        efe = sum_term1 + sum_term2 + sum_term3  # shape: [batch_size]
        # efe = efe * 50 # optionally multiply the efe to improve the loss magnitude for training the policy 
        efe_terms = [sum_term1, sum_term2, sum_term3]
            
        return efe, efe_terms
    


    def efe_full(self, o, calc_mean=False, samples=10):

        """
        Estimate Expected Free Energy with full MC sampling similar to DAIMC.
        """
        
        self.train()  # make sure dropout is enabled for MC sampling

        batch_size = o.size(0)
        device = o.device
        state_dim = self.encoder(o).size(1) // 2

        sum_term1 = torch.zeros(batch_size, 1, dtype=self.precision, device=device)
        sum_term2 = torch.zeros(batch_size, 1, dtype=self.precision, device=device)
        sum_term3 = torch.zeros(batch_size, 1, dtype=self.precision, device=device)

        qs0 = self.encoder(o)
        pi, ref = self.policy(o)
        qs0_s = reparameterize(qs0)
        qs0_s_ref = torch.cat([qs0_s, ref], dim=1)

        for i in range(samples):
            
            # θ ~ Q(θ|π), s ~ Q(s|o)
            psh = self.transition(qs0_s_ref)
            psh_logvar = psh[:, state_dim:]
            psh_s = reparameterize(psh)
            poh = self.decoder(psh_s)

            # sample o ~ P(o|s,θ)
            poh_s = po_max_sampler(poh, bcap=self.bcap, num_machines=self.action_dim-1)  # sample from the decoder output
            qsh = self.encoder(poh_s)
            qsh_logvar = qsh[:, state_dim:]

            # term 1: -log P(o|π)
            term1 = compute_term1(poh)

            # term 2: E[ H(s|o,π) - H(s|π) ]
            H_s_pi = entropy_normal_from_logvar(psh_logvar)
            H_s_o_pi = entropy_normal_from_logvar(qsh_logvar)
            term2 = torch.mean(H_s_o_pi - H_s_pi, dim=1, keepdim=True)

            # term 3: E[ H(o|s,θ,π) - H(o|s,π) ]
            term3_accum = torch.zeros_like(term1)
            for j in range(samples):
                
                # sample new θ and s
                psh_new = self.transition(qs0_s_ref)
                psh_s_new = reparameterize(psh_new)
                poh_temp1 = self.decoder(psh_s_new)
                term3_1 = entropy_bernoulli(poh_temp1)

                # same θ, new s (reparameterization trick)
                psh_s_temp2 = reparameterize(psh)
                poh_temp2 = self.decoder(psh_s_temp2)
                term3_2 = entropy_bernoulli(poh_temp2)

                term3 = torch.mean(term3_1 - term3_2, dim=1, keepdim=True)
                term3_accum += term3

            term3 = term3_accum / samples

            # accumulate all terms
            sum_term1 += term1
            sum_term2 += term2
            sum_term3 += term3

        # average across outer samples
        sum_term1 /= samples
        sum_term2 /= samples
        sum_term3 /= samples

        efe = sum_term1 + sum_term2 + sum_term3
        efe_terms = [sum_term1, sum_term2, sum_term3]

        return efe, efe_terms


       
         


       
        
            
    



    
