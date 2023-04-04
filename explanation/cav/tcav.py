import numpy as np
from tqdm import tqdm
import time
from torchvision.datasets import ImageNet

# from multiprocessing import dummy as multiprocessing
from explanation.cav.cav import CAV, get_or_train_cav
from explanation.cav.gradient_generator import GradientGenerator
from explanation.cav.activation_generator import ActivationGenerator


IMAGENET_DATASET_PATH = "F:\\ImageNet"

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

class TCAVRunParams:
    def __init__(self, 
                bottleneck,
                concept,
                target_class,
                activation_generator:ActivationGenerator,
                cav_dir,
                model,
                overwrite=True):
        self.bottleneck = bottleneck
        self.concept = concept
        self.target_class = target_class
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.overwrite = overwrite
        self.model = model


class TCAV:
    def __init__(self, 
                target,
                concepts,
                bottlenecks,
                activation_generator,
                alphas,
                random_counterpart=None,
                cav_dir=None,
                num_random_exp=5,
                random_concepts=None):


    def get_params(self):
        params = []
        for bottleneck in self.bottlenecks:
            for target_in_test, concepts_in_test in self.pairs_to_test:
                for alpha in self.alphas:
                  print('%s %s %s %s', bottleneck, concepts_in_test,
                                  target_in_test, alpha)
                  params.append({"bottleneck":bottleneck,"concepts":concepts,"target_class":target_class,"activation_generator":activation_generator,"cav_dir":cav_dir:,"overwrite":overwrite,"model":model})

        return params

    def _run_single_set(self, param:TCAVRunParams):
        model = param.model
        target_class = param.target_class
        concept = [param.concept]
        bottleneck = param.bottleneck
        cav_dir = param.cav_dir
        overwrite = param.overwrite
        activation_generator = param.activation_generator

        print('running %s %s' % (target_class, concept[0]))

        # Get acts
        acts = activation_generator.process_and_load_activations(bottlenecks, concept+[target])

        # Get CAVs
        cav_instance = get_or_train_cav(
                        concept,
                        bottleneck,
                        acts,
                        cav_dir=cav_dir,
                        cav_hparams=None,
                        overwrite=overwrite)

        # clean up
        for c in concept:
            del acts[c]

        # Hypo testing

        target_class_for_compute_tcav_score = target

        cav_concept = concept

        class_dir = ImageNet(IMAGENET_DATASET_PATH, download=False).class_to_idx
        class_id = class_dir[target_class_for_compute_tcav_score]

        i_up = compute_tcav_score(
            model, 
            class_id,
            cav_concept,
            cav_instance, 
            activation_generator.get_examples_for_concept(target),
            )

        print("i_up", i_up)

        val_directional_dirs = get_directional_dir(
            model,
            class_id,
            cav_concept,
            cav_instance,
            activation_generator.get_examples_for_concept(target)
            )

        print("val_directional_dirs", val_directional_dirs)

        cav_hparams = CAV.default_hparams()
        a_cav_key = CAV.cav_key(concept, bottleneck, cav_hparams['model_type'], cav_hparams['alpha'])

        result = {
            'cav_key':
                a_cav_key,
            'cav_concept':
                cav_concept,
            'negative_concept':
                concepts[1],
            'target_class':
                target_class,
            'cav_accuracies':
                cav_instance.accuracies,
            'i_up':
                i_up,
            'val_directional_dirs_abs_mean':
                np.mean(np.abs(val_directional_dirs)),
            'val_directional_dirs_mean':
                np.mean(val_directional_dirs),
            'val_directional_dirs_std':
                np.std(val_directional_dirs),
            'val_directional_dirs':
                val_directional_dirs,
            'bottleneck':
                bottleneck
        }

        del acts

        return result

    def run(self):
        results = []
        now = time.time()

        for i, param in enumerate(self.params):
            print('Running param %s of %s' % (i, len(self.params)))
            results.append(self._run_single_set(param, overwrite=overwrite))
        print('Done running %s params. Took %s seconds...' % (len(self.params), time.time() - now))

        return results
