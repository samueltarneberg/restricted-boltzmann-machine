import numpy as np


def XOR_function(n_inputs):
    functions =[[]]
    for i in range(n_inputs):
        new_functions = []
        for function in functions:
            new_functions.append(function + [0])
            new_functions.append(function + [1])
        functions = new_functions
    for function, j in zip(functions,range(len(functions))):
            XOR = 0
            for i in function:
                XOR = XOR ^ i
            functions[j].append(XOR)
            functions[j] = list(map(lambda bool: -1 if bool == 0 else bool, function))
    return functions



# def boltzmann_perceptron():
#      pass



def activation_probabilty(neuron_activation):
     return (1+np.tanh(neuron_activation)) / 2

def sample_neuron_state(probabilities):
    neuron_states =  []
    for probability in probabilities:
        random_number = np.random.rand()
        if random_number < probability:
            neuron_states.append(1)
        else:
            neuron_states.append(-1)
    return neuron_states

def train_restricted_Boltzmann_machine(num_epochs, 
                                       training_patterns, 
                                       pattern_probabilities, 
                                       num_hidden_neurons, 
                                       batch_size, 
                                       learning_rate, 
                                       gibs_steps):
    
    weight_matrix = np.random.randn(num_hidden_neurons,3)
    visible_thresholds = np.zeros(3)
    hidden_thresholds = np.zeros(num_hidden_neurons)

    for epoch_number in range(num_epochs):
        sample_indices = np.random.choice(len(training_patterns), batch_size, replace=True, p=pattern_probabilities)
        sample_patterns = np.array([training_patterns[index] for index in sample_indices])
        weight_gradient_sum, visible_bias_gradient_sum, hidden_bias_gradient_sum = np.zeros_like(weight_matrix), np.zeros_like(visible_thresholds), np.zeros_like(hidden_thresholds)

        for initial_visible_state in sample_patterns:
            initial_hidden_activations = np.dot(initial_visible_state, np.transpose(weight_matrix)) - hidden_thresholds # hidden_neruron b
            initial_hidden_probabilities = activation_probabilty(initial_hidden_activations)
            hidden_state = sample_neuron_state(initial_hidden_probabilities)

            for t in range(gibs_steps):
                visible_activations = np.dot(hidden_state, weight_matrix) - visible_thresholds
                visible_probabilities = activation_probabilty(visible_activations)
                reconstructed_visible_state = sample_neuron_state(visible_probabilities)
                
                reconstructed_hidden_activation = np.dot(reconstructed_visible_state, np.transpose(weight_matrix)) - hidden_thresholds
                reconstructed_hidden_probabilities = activation_probabilty(reconstructed_hidden_activation)
                hidden_state = sample_neuron_state(reconstructed_hidden_probabilities)

            weight_gradient_sum  += np.outer((np.tanh(initial_hidden_activations)), initial_visible_state) - np.outer((np.tanh(reconstructed_hidden_activation)), visible_activations)
            visible_bias_gradient_sum -= (initial_visible_state - reconstructed_visible_state)
            hidden_bias_gradient_sum -= (np.tanh(initial_hidden_activations)) - np.tanh(reconstructed_hidden_activation)

        weight_matrix += learning_rate * weight_gradient_sum
        visible_thresholds += learning_rate * visible_bias_gradient_sum
        hidden_thresholds += learning_rate * hidden_bias_gradient_sum

    return weight_matrix, visible_thresholds, hidden_thresholds

def sample_visible_state(weight_matrix, visible_thresholds, hidden_thresholds, visible_state):
        
        hidden_activation = np.dot(visible_state, np.transpose(weight_matrix)) - hidden_thresholds
        hidden_probabilities = activation_probabilty(hidden_activation)
        hidden_state = sample_neuron_state(hidden_probabilities)

        visible_activations = np.dot(hidden_state, weight_matrix) - visible_thresholds
        visible_probabilities = activation_probabilty(visible_activations)
        new_visible_state = sample_neuron_state(visible_probabilities)
        
        return new_visible_state


def estimate_model_probabilities(weight_matrix,visible_thresholds,hidden_thresholds, patterns, burn_in_steps, number_of_sampling_steps):
        
        visible_state = np.array([-1, 1, 1]) 
        for step in range(burn_in_steps):
            visible_state = sample_visible_state(weight_matrix, visible_thresholds, hidden_thresholds,visible_state)

        pattern_counts = {}
        number_of_sampling_steps = 1000000
        for step in range(number_of_sampling_steps):
            visible_state = sample_visible_state(weight_matrix, visible_thresholds, hidden_thresholds, visible_state)
            pattern = tuple(visible_state)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        probability_distribution = []

        for pattern in patterns:
            count = pattern_counts.get(tuple(pattern), 0)
            probability = count/number_of_sampling_steps
            probability_distribution.append(probability)
        
        return probability_distribution, pattern_counts
            
def kl_divergence(data_probabilities, model_probabilities):
     
    kl_divergence = 0

    for data_probability, model_probabilitiy in zip(data_probabilities, model_probabilities):
        if data_probability == 0:
            continue
        if model_probabilitiy == 0:
            return float("inf")
        
        kl_divergence += data_probability * np.log(data_probability/model_probabilitiy)
    return kl_divergence
                



def run(hidden_layer_sizes):
    
    patterns = [[-1, -1, -1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1],
            [ 1, -1,  1],
            [ 1, -1, -1],
            [ 1,  1,  1],
            [ 1,  1, -1]]

    probabilities = [0.25, 
                        0, 
                        0.25, 
                        0,
                        0.25, 
                        0, 
                        0.25, 
                        0]

    p0=10
    number_of_epochs = 10000
    eta = 0.0005
    k = 10
    burn_in_steps = 10_000
    number_of_sampling_steps = 2_000_000

    results = {}
    best_results = {}
    for num_hidden_neurons in hidden_layer_sizes:

        kl_values = []

        for run_number in range(10):
            weight_matrix, visible_thresholds, hidden_thresholds = train_restricted_Boltzmann_machine(number_of_epochs, patterns, probabilities, num_hidden_neurons, p0, eta, k)
        
            model_probabilities, pattern_counts = estimate_model_probabilities(weight_matrix, visible_thresholds, hidden_thresholds, patterns, burn_in_steps, number_of_sampling_steps)

            kl_divergence_value = kl_divergence(probabilities, model_probabilities)

            kl_values.append(kl_divergence_value)

            print(model_probabilities, pattern_counts)

        results[num_hidden_neurons] = kl_values

        for num_hidden_neurons, kl_values in results.items():
            best_results[num_hidden_neurons] = min(kl_values)
    # Count the visible state here
run([1,2,3,4,8])




                  
                
            





            
             
     
     

# initialise weights and thresholds;
# for ν= 1, . . . , νmax do
# sample p0 patterns from the data distribution (p0 ≤p );
# for µ= 1, . . . , p0 do
# initialise v (0)←x (µ);
# update all hidden neurons: b (h)(0)← v (0)−θ(h);
# for i= 1, . . . , M do
# hi (0)←+1 with probability p b (h)
# i (0) otherwise hi (0)←−1;
# end for
# for t = 1, . . . , k do
# update all visible neurons: b (v)(t−1)←h (t−1)· −θ(v);
# for j= 1, . . . , N do
# vj (t )←+1 with probability p b (v)
# j (t−1) otherwise vj (t )←−1;
# end for
# update all hidden neurons: b (h)(t )← v (t )−θ(h);
# for i= 1, . . . , M do
# hi (t )←+1 with probability p b (h)
# i (t ) otherwise hi (t )←−1;
# end for
# end for
# compute weight and theshold increments:
# δw (µ)
# mn
# ←η tanh b (h)
# m (0) vn (0)−tanh b (h)
# m (k ) vn (k ) ;
# δθ(v,µ)
# n
# ←−η[vn (0)−vn (k )];
# δθ(h,µ)
# m
# ←−ηtanh b (h)
# m (0)−tanh b (h)
# m (k ) ;
# end for
# adjust weights and thresholds using p0
# µ=1 δw (µ)
# mn ,
# p0
# µ=1 δθ(v,µ)
# n , and p0
# µ=1 δθ(h,µ)
# m ;
# end for
