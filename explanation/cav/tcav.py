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
                concepts,
                target_class,
                activation_generator:ActivationGenerator,
                cav_dir,
                model,
                overwrite=True):
        self.bottleneck = bottleneck
        self.concepts = concepts
        self.target_class = target_class
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.overwrite = overwrite
        self.model = model

    def __str__(self) -> str:
        return f"[bottleneck:{self.bottleneck},concepts:{self.concepts},target_class:{self.target_class}]"

class TCAV:
    def __init__(self, 
                target:str,
                concepts:list,
                bottlenecks,
                activation_generator:ActivationGenerator,
                cav_dir=None,
                num_random_exp=5):
        """Initialze tcav class.
            Args:
                sess: tensorflow session.
                target: one target class
                concepts: A list of names of positive concept sets.
                bottlenecks: the name of a bottleneck of interest.
                activation_generator: an ActivationGeneratorInterface instance to return
                                        activations.
                alphas: list of hyper parameters to run
                cav_dir: the path to store CAVs
                random_counterpart: the random concept to run against the concepts for
                              statistical testing. If supplied, only this set will be
                              used as a positive set for calculating random TCAVs
                num_random_exp: number of random experiments to compare against.
                
        """
        self.target = target
        self.concepts = concepts
        self.bottlenecks = bottlenecks
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.mymodel = activation_generator.model
        self.num_random_exp = num_random_exp

        self._process_what_to_run_expand(num_random_exp)
        self.params = self.get_params()
        print(('TCAV will %s params' % len(self.params)))


    def _process_what_to_run_expand(self, num_random_exp=100, random_concepts=None):
        """Get tuples of parameters to run TCAV with.

            TCAV builds random concept to conduct statistical significance testing
            againts the concept. To do this, we build many concept vectors, and many
            random vectors. This function prepares runs by expanding parameters.

        Args:
            num_random_exp: number of random experiments to run to compare.
            random_concepts: A list of names of random concepts for the random experiments
                        to draw from. Optional, if not provided, the names will be
                        random500_{i} for i in num_random_exp.
        """
        def get_random_concept(i):
            return (random_concepts[i] if random_concepts
                    else 'random500_{}'.format(i))

        """
        target_concept_pairs
        [
            [target1, [concept1]],
            [target1, [concept2]],
            ...,
        ]
        """
        target_concept_pairs = []
        for concept in self.concepts:
            target_concept_pairs.append([self.target,concept])

        """
        pairs_to_run_concepts
        [
            (t1, [c1, random1]),
            (t1, [c1, random2]),
            ...
            (t1, [c2, random1]),
            (t1, [c2, random2]),
            ...
        ]
        """
        pairs_to_run_concepts = []
        all_concepts = []
        for (target, concept) in target_concept_pairs:
            new_pairs_to_test_t = []
            # if only one element was given, this is to test with random.
    
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept != get_random_concept(i):
                    new_pairs_to_test_t.append((target, [concept, get_random_concept(i)]))
                    all_concepts.extend([concept, get_random_concept(i)])
                i += 1

            pairs_to_run_concepts.extend(new_pairs_to_test_t)
        all_concepts.append(self.target)
        all_concepts = list(set(all_concepts))

        # TODO random500_1 vs random500_0 is the same as 1 - (random500_0 vs random500_1)
        pairs_to_run_randoms = []
        all_concepts_randoms = []
        for i in range(num_random_exp):
            all_concepts_randoms.append(get_random_concept(i))
            for j in range(num_random_exp):
                if i!=j:
                    pairs_to_run_randoms.append((self.target,[get_random_concept(i),get_random_concept(j)]))
        
        self.all_concepts = list(set(all_concepts + all_concepts_randoms))
        self.pairs_to_test = pairs_to_run_concepts + pairs_to_run_randoms


    def get_params(self) -> list:
        params = []
        for bottleneck in self.bottlenecks:
            for target_in_test, concepts_in_test in self.pairs_to_test:
                # (t1, [c2, random2])
                print(bottleneck, concepts_in_test, target_in_test)
                params.append(TCAVRunParams(
                                        bottleneck,
                                        concepts_in_test,
                                        target_in_test,
                                        self.activation_generator,
                                        self.cav_dir,
                                        self.mymodel))

        return params

    def _run_single_set(self, param:TCAVRunParams):
        model = param.model
        target_class = param.target_class
        concepts = param.concepts
        bottleneck = param.bottleneck
        cav_dir = param.cav_dir
        overwrite = param.overwrite
        activation_generator = param.activation_generator

        print('running %s %s' % (target_class, concepts[0]))

        # Get acts
        acts = activation_generator.process_and_load_activations([bottleneck], concepts+[target_class])

        # Get CAVs
        cav_instance = get_or_train_cav(
                        concepts,
                        bottleneck,
                        acts,
                        cav_dir=cav_dir,
                        cav_hparams=None,
                        overwrite=overwrite)

        # clean up
        for c in concepts:
            del acts[c]

        # Hypo testing
        cav_concept = concepts[0]

        class_dir = ImageNet(IMAGENET_DATASET_PATH, download=False).class_to_idx
        class_id = class_dir[target_class]

        i_up = compute_tcav_score(
            model, 
            class_id,
            cav_concept,
            cav_instance, 
            activation_generator.get_examples_for_concept(target_class),
            )

        print("i_up", i_up)

        val_directional_dirs = get_directional_dir(
            model,
            class_id,
            cav_concept,
            cav_instance,
            activation_generator.get_examples_for_concept(target_class)
            )

        print("val_directional_dirs", val_directional_dirs)

        cav_hparams = CAV.default_hparams()
        a_cav_key = CAV.cav_key(concepts[0], bottleneck, cav_hparams['model_type'], cav_hparams['alpha'])

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

    def run(self, run_parallel=False):
        print(f"Running {self.params} params")

        results = []
        now = time.time()

        # if run_parallel:
        #     pool = multiprocessing.Pool()
        #     for i, res in enumerate(pool.imap(
        #         lambda p: self._run_single_set(p, run_parallel=run_parallel),
        #         self.params), 1):
        #         print('Finished running param %s of %s' % (i, len(self.params)))
        #         results.append(res)
        #     pool.close()
        # else:
        for i, param in enumerate(self.params):
            print('Running param %s of %s' % (i, len(self.params)))
            results.append(self._run_single_set(param))
        print('Done running %s params. Took %s seconds...' % (len(self.params), time.time() - now))

        return results
