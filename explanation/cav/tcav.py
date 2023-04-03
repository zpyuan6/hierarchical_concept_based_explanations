import numpy as np
from tqdm import tqdm
# from multiprocessing import dummy as multiprocessing
from explanation.cav.cav import CAV
from explanation.cav.gradient_generator import GradientGenerator

def get_direction_dir_sign(cav:CAV, concept, class_id, example, gradient_generator:GradientGenerator):
    """Get the sign of directional derivative.
    Args:
        mymodel: a model class instance
        act: activations of one bottleneck to get gradient with respect to.
        cav: an instance of cav
        concept: one concept
        class_id: index of the class of interest (target) in logit layer.
        example: example corresponding to the given activation
    Returns:
        sign of the directional derivative
    """
    # Get gradient
    gradient = gradient_generator.get_gradient(cav.bottleneck,example, class_id)

    # Grad points in the direction which DECREASES probability of class
    grad = np.reshape(gradient, -1)
    dot_prod = np.dot(grad, cav.get_direction(concept))

    return dot_prod < 0

def compute_tcav_score(model,
                        class_id,
                        concept,
                        cav:CAV,
                        examples,
                        run_parallel=True):
    """Compute TCAV score.
    Args:
      model: a model class instance
      target_class: one target class
      concept: one concept
      cav: an instance of cav
      class_acts: activations of the examples in the target class where
        examples[i] corresponds to class_acts[i]
      examples: an array of examples of the target class where examples[i]
        corresponds to class_acts[i]
      run_parallel: run this parallel fashion
      num_workers: number of workers if we run in parallel.
    Returns:
        TCAV score (i.e., ratio of pictures that returns negative dot product
        wrt loss).
    """

    count = 0

    # Get gradient
    gradient_generator = GradientGenerator(model)

    # Waiting deployment, due to thread safe
    # if run_parallel:
    #     pool = multiprocessing.Pool()
    #     directions = pool.map(
    #         lambda i: get_direction_dir_sign(model, np.expand_dims(class_acts[i], 0), cav, concept, class_id, examples[i]), range(len(class_acts)),gradient_generator)
    #     pool.close()
    #     return sum(directions) / float(len(class_acts))
    # else:
    with tqdm(total=len(examples)) as tbar:
        for i in range(len(examples)):
            example = examples[i]
            if get_direction_dir_sign(cav, concept, class_id, example, gradient_generator):
                count += 1
            tbar.update(1)
            
          
    return float(count) / float(len(examples))

def get_directional_dir(mymodel, class_id, concept, cav:CAV, examples):
    """Return the list of values of directional derivatives.
       (Only called when the values are needed as a referece)
    Args:
      mymodel: a model class instance
      class_id: index of one target class
      concept: one concept
      cav: an instance of cav
      class_acts: activations of the examples in the target class where
        examples[i] corresponds to class_acts[i]
      examples: an array of examples of the target class where examples[i]
        corresponds to class_acts[i]
    Returns:
      list of values of directional derivatives.
    """
    directional_dir_vals = []

    # Get gradient
    gradient_generator = GradientGenerator(mymodel)

    with tqdm(total=len(examples)) as tbar:
        for i in range(len(examples)):
            example = examples[i]
            gradient = gradient_generator.get_gradient(cav.bottleneck, example, class_id)
            grad = np.reshape(gradient, -1)
            directional_dir_vals.append(np.dot(grad, cav.get_direction(concept)))

            tbar.update(1)

    return directional_dir_vals