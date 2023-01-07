from utils.MemoryKernel import ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, \
                                GammaGompertzMemoryKernel, WeibullMemoryKernel,ExponentialPseudoMemoryKernel,GompertzMemoryKernel



######################################### Global parameters ##########################################
# Save every n iteration
n_save_iter = 5

# debuggig flag to test an individual sub problem
run_test_sub_problem = False
test_sub_problem = 32
visualize_subproblems = False


# Initial value of alpha values before starting training
# Note: this has to be kept relatively small w.r.t to the psi function since if it is too large, 
# the gradient updates will be very small
#init_alpha = 1.0



# optimization algorithm
opt_alg = "bpgd"

reinitialize = False
run_setup = False
train = False
evaluate = True
    
dataset = "synthetic"

start_size_list = [5,7,10,12]
    
if dataset == "synthetic":
    ################################################ synthetic ################################################
    # APGG algorithm
    t_max = 100
    users_to_run = "all"
    scenario = "synthetic"
    inner_iter = 40
    init_alpha = 1.0e-3
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 30
    tc_list = [0.1,0.15,.25,0.3]

    
    
    memory_kernel = ExponentialPseudoMemoryKernel(0.05)
    run_name = "ExponentialPseudoMemoryKernel0.05"

    # ell-1 regularization parameters
    gamma = 0.0

    rho = 10
    beta_u = 1.2
    beta_d = 1.4 # does not play role for bpgd
    L0 = 1.0
    epsilon = 10e-15
    tol = 10e-15

    # consensus ADMM algorithm
    mu = 1.2
    tau_incr = 1.4
    tau_decr = 1.8
    
elif dataset == "irvine":
    ################################################ Irvine ################################################
    # APGG algorithm
    t_max = 20
    users_to_run = "all"
    scenario = "irvine"
    inner_iter = 10
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10
    init_alpha = 1.0e-2
    tc_list = [1*24,2*24,30*24,60*24]

    
    memory_kernel = PowerLawMemoryKernel(0.5)
    run_name = "PowerLawMemoryKernel0.5"
    
    #memory_kernel = ExponentialPseudoMemoryKernel(0.5)
    #run_name = "ExponentialPseudoMemoryKernel(0.5)"
    
    # ell-1 regularization parameters
    gamma = 0.5


    rho = 10
    beta_u = 1.5
    beta_d = 1.4 # does not play role for bpgd
    L0 = 100.0
    epsilon = 10e-5
    tol = 10e-10

    # consensus ADMM algorithm
    mu = 10
    tau_incr = 3.0
    tau_decr = 1.2
    
elif dataset == "lastfm":
    #### Lastfm ###
    t_max = 20
    users_to_run = "all"
    inner_iter = 5
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10
    init_alpha = 0.01
    tc_list = [1*24,2*24,30*24,60*24]


    scenario = "lastfm"
    memory_kernel = PowerLawMemoryKernel(1.0)
    run_name = "PowerLawMemoryKernel1.0"
    
    #memory_kernel = ExponentialPseudoMemoryKernel(0.5)
    #run_name = "ExponentialPseudoMemoryKernel0.5"

    #memory_kernel = ConstantMemoryKernel()
    #run_name = "ConstantMemoryKernel0.0"

    # ell-1 regularization parameters
    gamma = 5.0


    # sub problem optimizer hyper-parameter    
    rho = 10.0
    beta_u = 2.5
    beta_d = 1.4 # does not play role for bpgd
    L0 = 10.0
    epsilon = 10e-7
    tol = 10e-7

    # consensus ADMM algorithm
    mu = 10.0
    tau_incr = 2.0
    tau_decr = 2.0

elif dataset == "github":
    ### github ###
    t_max = 20
    users_to_run = "all"
    scenario = "github"
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10

    memory_kernel = PowerLawMemoryKernel(1.0)
    run_name = "PowerLawMemoryKernel1.0"
    inner_iter = 30
    
    # ell-1 regularization parameters
    gamma = 1.0


    # consensus ADMM hyper-parameter  
    rho = 10000
    tol = 10e-4
    beta_u = 2.2
    beta_d = 2.5
    L0 = 100
    epsilon = 10e-5
    

    mu = 1.5
    tau_incr = 1.1
    tau_decr = 1.5

elif dataset == "twitter":
    ### Twitter ###
    t_max = 100
    users_to_run = "all"
    scenario = "twitter"
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10


    memory_kernel = ExponentialPseudoMemoryKernel(1.0)
    run_name = "ExponentialPseudoMemoryKernel1.0"
    inner_iter = 30
    
    # consensus admm parameters
    rho = 100
    beta_u = 1.5
    beta_d = 1.6
    L0 = 100
    epsilon = 10e-5
    tol = 10e-10

    # ell-1 regularization parameters
    gamma = 1.0



    # consensus ADMM algorithm
    mu = 10.0
    tau_incr = 3.0
    tau_decr = 1.5
    
elif dataset == "lastfm_links":
    ### Lastfm links###
    t_max = 1
    users_to_run = "all"
    scenario = "lastfm_link"
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10

    memory_kernel = PowerLawMemoryKernel(0.15)
    run_name = "PowerLawMemoryKernel0.15"
    inner_iter = 30
    
    # consensus ADMM hyper-parameter    
    rho = 1
    tol = 10e-4
    mu = 1.1
    tau_incr = 1.2
    tau_decr = 1.1

    # ell-1 regularization parameters
    gamma = 1.0


    # Scheibergs accelerated proximal gradient descent
    epsilon = 1.e-3
    beta_u = 5.5
    beta_d = 1.5
    L0 = 100

elif dataset == "digg":
    ### Digg ###
    t_max = 100
    users_to_run = "all"
    inner_iter = 15
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 20
    init_alpha = 1.e-3
    tc_list = [1,24,36,72]
    
    
    scenario = "digg"
    #memory_kernel = PowerLawMemoryKernel(0.5)
    #run_name = "PowerLawMemoryKernel0.5"
    
    
    memory_kernel = ExponentialPseudoMemoryKernel(0.5)
    run_name = "ExponentialPseudoMemoryKernel0.5"
    
    #memory_kernel = ConstantMemoryKernel()
    #run_name = "ConstantMemoryKernel0.0"

    # ell-1 regularization parameters
    gamma = 1.0


    rho = 0.1
    beta_u = 1.4
    beta_d = 1.4 # does not play role for bpgd
    L0 = 10.0
    epsilon = 10e-5
    tol = 10e-5

    # consensus ADMM algorithm
    mu = 1.5
    tau_incr = 1.5
    tau_decr = 1.6
    
elif dataset == "memes":

    ### memes ###
    t_max = 100
    users_to_run = "all"
    inner_iter = 5
    init_alpha = 1.0
    # Number of iterations of accelerated nesterov's projected gradient descent 
    sub_problem_iterations = 10
    tc_list = [0.2,0.4,1,2]
    
    
    scenario = "memes"
    #memory_kernel = PowerLawMemoryKernel(0.0005)
    #run_name = "PowerLawMemoryKernel0.0005"
    
    memory_kernel = ExponentialPseudoMemoryKernel(0.5)
    run_name = "ExponentialPseudoMemoryKernel0.5"
    
    #memory_kernel = ConstantMemoryKernel()
    #run_name = "ConstantMemoryKernel0.0"

    # ell-1 regularization parameters
    gamma = 0.0

    # consensus ADMM hyper-parameter    
    rho = 0.1
    beta_u = 1.3
    beta_d = 1.4 # does not play role for bpgd
    L0 = 1.0
    epsilon = 10e-7
    tol = 10e-7

    # consensus ADMM algorithm
    mu = 1.5
    tau_incr = 1.3
    tau_decr = 1.2

