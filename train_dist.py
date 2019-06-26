# from maskrcnn.set_maskrcnn_config_file import set_maskrcnn_config_file  
from tools import Options, distributed_train

class Distributed(object):
    def __init__(self, args):
        # assert args.training_script_args is not None
        self.args = args
        print(f"Dist script: {args.script}")
        print(f"Dist script args: {args.script_args}")
        
    def train(self):        
        distributed_train(self.args)

if __name__ == "__main__":

    # maskrcnn_config_file = set_maskrcnn_config_file()

    options = Options()
    # options.training_script_args = [
    #     '--config-file', '%s' %maskrcnn_config_file
    #     ]
    options.nproc_per_node = 4

    model = Distributed(options)
    model.train()