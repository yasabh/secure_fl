import os
import torch
import subprocess
import attacks
from mpspdz.ExternalIO import mpc_client

class MPC:

    def __init__(self, net, protocol, aggregation, num_parties, port, niter, learning_rate, chunk_size, num_clients, byz, num_byz, threads, parallels, always_compile):
        self.net = net
        self.port = port
        self.niter = niter
        self.learning_rate = learning_rate
        self.chunk_size = chunk_size
        self.byz = byz
        self.num_byz = num_byz
        self.threads = threads
        self.parallels = parallels
        self.always_compile = always_compile
        self.script, self.num_parties = self.get_protocol(protocol, num_parties)
        self.server_process = None
        self.device = None

        if aggregation == "fedavg":
            self.filename_server = "mpc_fedavg_server"
            self.num_clients = num_clients
        elif aggregation == "fltrust":
            self.filename_server = "mpc_fltrust_server"
            self.num_clients = num_clients + 1
        else:
            raise NotImplementedError

        self.num_params = torch.cat([xx.reshape((-1, 1)) for xx in self.net.parameters()], dim=0).size()[0]
        self.full_filename = f'{self.filename_server}-{self.port}-{self.num_params}-{self.num_clients}-{self.niter}-{self.chunk_size}-{self.threads}-{self.parallels}'
        # self.compile()

    def get_protocol(self, protocol, players):
        """
        Returns the shell script name and number of players for the protocol.
        protocol: name of the protocol
        players: number of parties
        """
        if players < 2:
            raise Exception("Number of players must at least be 2")

        if protocol == "semi2k":
            return "semi2k.sh", players

        elif protocol == 'spdz2k':
            return "spdz2k.sh", players

        elif protocol == "replicated2k":
            if players != 3:
                raise Exception("Number of players must be 3 for replicated2k")
            return "ring.sh", 3

        elif protocol == "psReplicated2k":
            if players != 3:
                raise Exception("Number of players must be 3 for psReplicated2k")
            return "ps-rep-ring.sh", 3

        else:
            raise NotImplementedError

    def compile(self):
        os.chdir("mpspdz")

        if not os.path.exists('./Programs/Bytecode'):
            os.mkdir('./Programs/Bytecode')
        already_compiled = len(list(filter(lambda f: f.find(self.full_filename) != -1, os.listdir('./Programs/Bytecode')))) != 0

        if self.always_compile or not already_compiled:
            # compile mpc program, arguments -R 64 -X were chosen so that every protocol works
            os.system('./compile.py -R 64 -X ' + self.filename_server + ' ' + str(self.port) + ' '
                      + str(self.num_params) + ' ' + str(self.num_clients) + ' ' + str(self.niter) + ' '
                      + str(self.chunk_size) + ' ' + str(self.threads) + ' ' + str(self.parallels))

        # setup ssl keys
        os.system('Scripts/setup-ssl.sh ' + str(self.num_parties))
        os.system('Scripts/setup-clients.sh 1')

        os.chdir("..")

    def run(self, device):
        self.device = device
        os.chdir("mpspdz")

        print("Starting Computation Parties")
        # start computation servers using a child process to run in parallel
        self.server_process = subprocess.Popen(["./run_aggregation.sh", self.script, self.full_filename, str(self.num_parties)])

        os.chdir("..")

    def wait(self):
        self.server_process.wait()

    def get_payload_size(self, gradients: list[torch.Tensor]) -> int:
        """
        Calculate the total byte size of all gradients combined across clients.
        Each client's gradients are flattened and concatenated.
        """
        param_list = [torch.cat([g.reshape((-1, 1)) for g in client], dim=0) for client in gradients]
        combined_tensor = torch.reshape(torch.cat(param_list, dim=0), (-1,))
        total_elements = combined_tensor.numel()
        element_size = combined_tensor.element_size()  # in bytes
        return total_elements * element_size

    def aggregate(self, gradients):
        """
        Secure aggregation using MP-SPDZ.
        gradients: list of gradients (each is a list of tensors
        """
        total_bytes = self.get_payload_size(gradients)
        print(f"Total gradient payload size: {total_bytes} bytes")

        param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
        if self.byz is not None:
            # let the malicious clients (first f clients) perform the byzantine attack
            param_list = self.byz(param_list, self.net, self.learning_rate, self.num_byz, self.device)

        n = len(param_list)
        param_num = sum(param.numel() for param in self.net.parameters())
        param_list_python = torch.reshape(torch.cat(param_list, dim=0), (-1,)).tolist()  # convert tensors to list

        os.chdir("mpspdz")
        output = mpc_client.client(0, self.num_parties, self.port, param_num, n, self.chunk_size, param_list_python, precision=12)
        os.chdir("..")

        global_update = torch.tensor(output).to(self.device)  # convert python list to tensor

        # update global model
        idx = 0
        for j, (param) in enumerate(self.net.parameters()):
            param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-self.learning_rate)
            idx += torch.numel(param)