import torch


class DefaultParams:
    """
    Do not change the default values. if you want different values, create a new instance and override
    """

    def __init__(self):
        self.attack = 'popskip'
        self.targeted = False
        self.dataset = 'mnist'
        self.model_keys: dict = {'mnist': ['mnist_noman'], 'cifar10': ['cifar10'], 'cifar100': ['cifar100'],
                                 'imagenet': ['imagenet']}
        self.model_keys_filepath = None

        self.num_iterations = 32

        self.internal_dtype = torch.float32
        self.bounds = (0, 1)
        self.gamma = 1.0
        self.input_image_path = None  # String
        self.input_image_label = None  # integer
        self.init_image_path = None
        self.hsja_repeat_queries = 1  # parameter for HSJ-repeated

        self.initial_num_evals = 100  # B_0 (i.e. num of queries for first iteration of original HSJA)
        self.max_num_evals = 50000  # Maximum queries allowed in Approximate Gradient Step
        self.eval_factor = 1  # times the number of queries in approx gradient step
        # self.stepsize_search = "geometric_progression"  # Deprecating this
        self.distance = "l2"  # Distance metric
        self.batch_size = 256

        # Hand-picking images
        self.orig_image_conf = 0.75
        # self.orig_image_conf = 0.95
        self.max_queries = 20000

        # thresholding
        self.threshold = 0.00

        # Specific to Noisy Models
        self.noise = 'bayesian'
        self.new_adversarial_def = True  # New Def: if p(true_label)<0.5 then its adversarial
        self.sampling_freq_binsearch = 1
        self.ask_human = False
        self.slack = 0.0
        self.flip_prob = 0.2  # Specific to Stochastic Noise
        self.beta = 1.0  # Gibbs Distribution Parameter (p ~ exp(beta*x))
        self.smoothing_noise = 0.01
        self.crop_size = 28
        self.drop_rate = 0.

        # Specific to Info max procedure
        self.grid_size = {'mnist': 100, 'cifar10': 300, 'cifar100': 300, 'imagenet': 1500}
        self.prior_frac = 1
        self.queries = 1
        self.infomax_stop_criteria = "estimate_fluctuation"

        # Specific to Approximate Gradient
        self.grad_queries = 1

        self.theta_fac = -1

        # Specific to Experiment mode
        self.experiment_mode = True
        self.num_samples = 3
        self.samples_from = 0  # Number of sample to skip. This is for merging
        self.experiment_name = None  # If not none it will override the command line argument
