import os
import torch
import numpy as np
from torch import nn
import torch.optim as topt
from torch.utils.tensorboard import SummaryWriter
import json
from circuit_ud_matrix import Circuit_manager

#TARGET = torch.Tensor(np.array([[1., 0., 0., 0., 0., 0., 0., 0.]
#                                   , [0., 1., 0., 0., 0., 0., 0., 0.]
#                                   , [0., 0., 1., 0., 0., 0., 0., 0.]
#                                   , [0., 0., 0., 1., 0., 0., 0., 0.]
#                                   , [0., 0., 0., 0., 1., 0., 0., 0.]
#                                   , [0., 0., 0., 0., 0., 1., 0., 0.]
#                                   , [0., 0., 0., 0., 0., 0., 0., 1.]
#                                   , [0., 0., 0., 0., 0., 0., 1., 0.]]))
TARGET = torch.eye(16)
LOSS_FACTOR = 1
class DQAS4RL:
    def __init__(self,
                 qdqn,
                 gamma,
                 lr,
                 lr_struc,
                 batch_size,
                 greedy,
                 greedy_decay,
                 greedy_min,
                 update_model,
                 update_targ_model,
                 memory_size,
                 max_steps,
                 seed,
                 cm: Circuit_manager,
                 prob_max=0.5,
                 lr_in=None,
                 lr_out=None,
                 loss_func='MSE',
                 opt='Adam',
                 opt_struc='Adam',
                 logging=False,

                 sub_batch_size=1,
                 early_stop=195,
                 structure_batch=10,

                 total_epochs=5000,
                 struc_learning=True,
                 struc_early_stop=0,
                 p_search=False,
                 p_search_lowerbound=2,
                 p_search_period=0,
                 min_loss=0.1):

        self.qdqn = qdqn
        self.cm = cm

        self.gamma = gamma
        self.lr = lr
        self.lr_struc = lr_struc
        self.lr_in = lr_in
        self.lr_out = lr_out

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seed = seed

        self.greedy = greedy
        self.greedy_decay = greedy_decay
        self.greedy_min = greedy_min

        self.update_model = update_model
        self.update_targ_model = update_targ_model

        self.loss_func_name = loss_func
        self.opt_name = opt
        self.opt_name_struc = opt_struc
        self.prob_max = prob_max
        self.avcost = 0
        self.early_stop = early_stop
        self.min_loss = min_loss

        self.struc_early_stop = struc_early_stop if struc_early_stop else int(total_epochs / (2 * (int(cm.num_placeholders / cm.learning_step))))
        self.struc_early_step = self.struc_early_stop
        self.struc_learning = struc_learning
        self.total_epochs = total_epochs

        self.p_search = p_search
        self.p_search_lowerbound = p_search_lowerbound
        self.p_search_period = p_search_period if p_search_period else int(
            self.struc_early_stop / (2 * self.cm.num_ops))

        self.device = 'cpu'
        self.logging = logging

        self.sub_batch_size = sub_batch_size
        self.structure_batch = structure_batch
        self.best_structs = []
        self.dtype = torch.float32  # TODO think this dtype

        self.config()

    def config(self):
        # device
        self.device = torch.device(self.device)

        # set network to device
        self.qdqn = self.qdqn.to(self.device)

        # initial struc parameters
        if self.struc_learning:
            v_struc_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            self.var_struc = torch.tensor(v_struc_init, requires_grad=True, dtype=self.dtype, device=self.device)

        # set loss functions
        self.loss_func = getattr(nn, self.loss_func_name + 'Loss')()
        torch_opt = getattr(topt, self.opt_name)
        if self.struc_learning:
            torch_opt_struc = getattr(topt, self.opt_name_struc)

        params = []
        params.append({'params': self.qdqn.q_layers.parameters()})

        # set optimizers
        self.opt = torch_opt(params, lr=self.lr)
        if self.struc_learning:
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc, betas=(0.9, 0.99), weight_decay=0.001)

        if self.logging:
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')
            out_path =  "_"  + "_"
            self.log_dir = f'./logs/{out_path}/'
            self.reprot_dir = f'./logs/{out_path}/reports/'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.reprot_dir):
                os.makedirs(self.reprot_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step = 0
        self.epoch_count = 0

    def matrix_loss_func(self, a, b):
        return torch.linalg.norm(b - a)

    def push_json(self, out, path):
        with open(path, 'w') as f:
            json.dump(out, f, indent=4)

    def preset_byprob(self, prob):
        return np.array([np.random.choice(np.arange(prob.size()[1])
                                          , p=np.array(prob[i].detach().cpu().clone().numpy())) for i in range(prob.size()[0])])

    def train_model(self, prob):

        self.qdqn.train()
        self.opt.zero_grad()

        # structure training
        loss_list = []
        deri_struc = []

        if self.prob_max and self.struc_learning:
            prob = torch.clamp(prob, min=(1 - self.prob_max) / self.cm.num_ops, max=self.prob_max)
            prob = torch.transpose(prob.t() / torch.sum(prob, dim=1), 0, 1)

        grad_params = {}
        if self.struc_learning:
            self.opt_struc.zero_grad()
            sum_sample_strucs = 0
            for idx in range(self.structure_batch):
                # -- sample struc --
                sample_struc = self.preset_byprob(prob)
                self.cm.set_current_sampled_struc(sample_struc)

                # -- predicted --
                states = torch.Tensor([0.] * self.cm.num_qubits)
                predicted = self.qdqn(states)

                # -- target --
                expected_qvalues = TARGET  # TODO original for ccnot

                # -- loss --
                sub_loss = self.matrix_loss_func(predicted, expected_qvalues) * LOSS_FACTOR
                sub_sum = sum(
                    map(self.cm.check_gate, [self.cm.ops[gate][0] for gate in sample_struc] + self.cm.sphc_struc))

                sum_sample_strucs += sub_sum
                if sub_sum:
                    sub_loss.backward()
                    ## -- gradient --
                    for name, param in self.qdqn.named_parameters():

                        if param.grad is None:
                            gr = torch.from_numpy(np.zeros(self.cm.get_weights_shape(), dtype=np.float32)).type(
                                self.dtype)
                        else:
                            gr = param.grad.detach().clone()
                        summary = grad_params.get(name, [])
                        summary.append(gr)
                        grad_params[name] = summary

                loss_list.append(sub_loss)
                with torch.no_grad():
                    grad_struc = (-prob).type(self.dtype).to(self.device).index_put(
                        indices=tuple(
                            torch.LongTensor(list(zip(range(self.cm.current_num_placeholders), sample_struc))).t())
                        , values=torch.ones([self.cm.current_num_placeholders], dtype=self.dtype, device=self.device)
                        , accumulate=True).to(self.device)
                    deri_struc.append(((sub_loss - self.avcost) * grad_struc).to(self.device))

            with torch.no_grad():
                total_loss = torch.mean(torch.stack(loss_list)) * LOSS_FACTOR
                self.avcost = torch.mean(torch.stack(loss_list)) * LOSS_FACTOR

            if sum_sample_strucs:
                with torch.no_grad():
                    for name, param in self.qdqn.named_parameters():
                        if name in grad_params.keys():
                            param.grad = torch.mean(torch.stack(grad_params[name]), dim=0).type(self.dtype).to(
                                self.device)

                self.opt.step()
            with torch.no_grad():
                grad_batch_struc = torch.mean(torch.stack(deri_struc), dim=0).type(self.dtype).to(self.device)
                self.var_struc.grad = grad_batch_struc

            self.opt_struc.step()
            # self.opt_beta.step()
            #predicted_for_show()
        else:
            self.qdqn.train()
            self.opt.zero_grad()
            # -- predicted --
            states = torch.Tensor([.0] * self.cm.num_qubits)
            predicted = self.qdqn(states)

            # -- target --
            expected_qvalues = TARGET
            total_loss = self.matrix_loss_func(predicted, expected_qvalues)
            total_loss.backward()

            self.opt.step()

        return total_loss.item()

    def update_prob(self):
        with torch.no_grad():
            prob = torch.zeros(self.var_struc.shape)
            for i in range(self.cm.current_num_placeholders):
                row_sum = torch.sum(torch.exp(self.var_struc[i][self.tmp_ops_candidates[i]]))
                for j, v in enumerate(self.var_struc[i]):
                    if j in self.tmp_ops_candidates[i]:
                        prob[i, j] = torch.exp(v) / row_sum
        return prob

    def epoch_train(self, epoch):
        epoch_reward = 0
        epoch_loss = []
        epoch_greedy = 0

        prob = self.update_prob()
        # train in batch
        if epoch % self.update_model == 0:
            loss = self.train_model(prob)
            epoch_loss.append(loss)

        self.epoch_count += 1
        epoch_avg_loss = np.mean(epoch_loss)
        self.push_json([*self.cm.get_current_layer_struc()], self.log_dir + 'current_struc.json')

        if epoch > self.struc_early_stop and self.struc_learning:
            # learning place and state updated
            prob = self.update_prob()
            #print(prob)
            layer_learning_finished = self.cm.update_learning_places(prob=prob)
            if layer_learning_finished:
                self.struc_learning = self.cm.learning_state
                if not self.struc_learning:  # learning finish
                    return {
                        'steps': epoch,
                        'avg_loss': epoch_avg_loss,
                        'reward': epoch_reward,
                        'greedy': epoch_greedy,
                        'prob': prob.detach().cpu().clone().numpy()
                    }
                else:
                    pass
            torch_opt_struc = getattr(topt, self.opt_name_struc)
            v_struc_update_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)

            self.var_struc = torch.tensor(v_struc_update_init, requires_grad=True, dtype=self.dtype, device=self.device)
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            self.struc_early_stop += self.struc_early_stop

        if self.struc_learning:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
                'reward': epoch_reward,
                'greedy': epoch_greedy,
                'prob': prob.detach().cpu().clone().numpy(),
               # 'pridected':
            }
        else:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
                'reward': epoch_reward,
                'greedy': epoch_greedy,
                #'pridected':
            }

    def learn(self):

        postfix_stats = {}
        records_reward = []
        records_steps = []
        records_avg_loss = []
        records_greedy = []
        records_probs = []
        avg_rewards = []
        norm_rewards = []
        self.push_json(self.cm.ops, self.log_dir + 'operation_pool')

        for t in range(self.total_epochs):
            # train dqn
            train_report = self.epoch_train(t)
            #print(train_report)
            #print(type(train_report))
            postfix_stats['train/epoch_loss'] = train_report['avg_loss']
            postfix_stats['train/epochs'] = train_report['steps']
            postfix_stats['strucl'] = self.struc_learning

            records_reward.append(train_report['reward'])
            records_steps.append(train_report['steps'])
            records_avg_loss.append(train_report['avg_loss'])
            records_greedy.append(train_report['greedy'])
            if self.struc_learning:
                records_probs.append(train_report['prob'].tolist())

            avg_rewards.append(np.mean(records_reward[-100:]))

            mean_reward = np.mean(records_reward)
            std_reward = np.std(records_reward)

            if std_reward != 0:
                norm_reward = (records_reward[-1] - mean_reward) / std_reward
                norm_rewards.append(norm_reward)
            else:
                norm_rewards.append(0)

            if records_avg_loss[-1] <= self.min_loss:
                output = f"problem solved at epoch {t}, with loss {records_avg_loss[-1]}"
                if self.struc_learning:
                    output += "learning state {self.struc_learning}"
                self.push_json(output, self.log_dir + 'problem_solved.json')
                print(output)
                break