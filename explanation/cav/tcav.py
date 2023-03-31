import numpy as np
from multiprocessing import dummy as multiprocessing
from cav import CAV

def get_direction_dir_sign(mymodel, act, cav:CAV, concept, class_id, example):
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
    # Grad points in the direction which DECREASES probability of class
    grad = np.reshape(mymodel.get_gradient(act, [class_id], cav.bottleneck, example), -1)
    dot_prod = np.dot(grad, cav.get_direction(concept))
    return dot_prod < 0

def compute_tcav_score(model,
                        class_id_dir,
                        target_class,
                        concept,
                        cav:CAV,
                        class_acts,
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
    class_id = class_id_dir[target_class]
    if run_parallel:
        pool = multiprocessing.Pool()
        directions = pool.map(
            lambda i: get_direction_dir_sign(model, np.expand_dims(class_acts[i], 0), cav, concept, class_id, examples[i]), range(len(class_acts)))
        pool.close()
        return sum(directions) / float(len(class_acts))
    else:
        for i in range(len(class_acts)):
            act = np.expand_dims(class_acts[i], 0)
            example = examples[i]
            if get_direction_dir_sign(model, act, cav, concept, class_id, example):
                count += 1
        return float(count) / float(len(class_acts))