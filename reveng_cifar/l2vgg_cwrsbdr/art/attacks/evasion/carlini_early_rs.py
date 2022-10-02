# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the L2, LInf and L0 optimized attacks `CarliniL2Method`, `CarliniLInfMethod` and `CarliniL0Method
of Carlini and Wagner (2016). These attacks are among the most effective white-box attacks and should be used among the
primary attacks to evaluate potential defences. A major difference with respect to the original implementation
(https://github.com/carlini/nn_robust_attacks) is that this implementation uses line search in the optimization of the
attack objective.

| Paper link: https://arxiv.org/abs/1608.04644
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.optimizers import Adam
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import (
    compute_success,
    get_labels_np_array,
    tanh_to_original,
    original_to_tanh,
)
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class CarliniL2Method(EvasionAttack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is among the most effective and should be used
    among the primary attacks to evaluate potential defences. A major difference wrt to the original implementation
    (https://github.com/carlini/nn_robust_attacks) is that we use line search in the optimization of the attack
    objective.

    | Paper link: https://arxiv.org/abs/1608.04644
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "max_halving",
        "max_doubling",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        binary_search_steps: int = 10,
        max_iter: int = 10,
        initial_const: float = 1,
        max_halving: int = 5,
        max_doubling: int = 5,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a Carlini&Wagner L_2 attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results
               but are slower to converge.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If
                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the
                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are
                                    hardcoded with the same values used by the authors of the method.
        :param max_iter: The maximum number of iterations.
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and
                confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
                Carlini and Wagner (2016).
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.initial_const = initial_const
        self.max_halving = max_halving
        self.max_doubling = max_doubling
        self.batch_size = batch_size
        self.verbose = verbose
        CarliniL2Method._check_params(self)

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10

        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero.
        # It appears this is what Carlini and Wagner (2016) are alluding to in their footnote 8. However, it is not
        # clear how their proposed trick ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.
        self._tanh_smoother = 0.999999

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the objective function value.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param target: An array with the target class (one-hot encoded).
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: A tuple holding the current logits, l2 distance and overall loss.
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        z_predicted = self.estimator.predict(
            np.array(x_adv, dtype=ART_NUMPY_DTYPE),
            logits=True,
            batch_size=self.batch_size,
        )
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(
            z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, np.zeros(x.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, np.zeros(x.shape[0]))

        return z_predicted, l2dist, c_weight * loss + l2dist

    def _loss_gradient(
        self,
        z_logits: np.ndarray,
        target: np.ndarray,
        x: np.ndarray,
        x_adv: np.ndarray,
        x_adv_tanh: np.ndarray,
        c_weight: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        :param z_logits: An array with the current logits.
        :param target: An array with the target class (one-hot encoded).
        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :param c_weight: Weight of the loss term aiming for classification as target.
        :param clip_min: Minimum clipping value.
        :param clip_max: Maximum clipping value.
        :return: An array with the gradient of the loss function.
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )

        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]

        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        return loss_gradient

    def random_start(self,x: np.ndarray, eps, clip_min,clip_max):
        nprd = np.random.RandomState()
        y_p = np.argmax( get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)))    
        for _ in range(100):
            print(clip_min,clip_max)
            random_img = nprd.uniform(clip_min,clip_max,size = x.shape).astype(x.dtype)
            #random_img = x+ nprd.uniform(-eps,eps, size=x.shape).astype(x.dtype)
            #random_img = np.clip(random_img,clip_min,clip_max)
            #print("random_img",type(random_img),np.shape(random_img))
            random_class = np.argmax(get_labels_np_array(self.estimator.predict(random_img, batch_size=self.batch_size)))
            #print("y_p",y_p)
            #print("random_class",random_class)
            #if random_class == y_p:
            if random_class != y_p:
             #logger.info("Found initial adversarial image for untargeted attack.")
                print("max random image",np.max(random_img))
                print("min random image", np.min(random_img))
                return random_img
                #break
        #print("no unsuccessful noise")
        return random_img
       
        

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="C&W L_2", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            #print("x_batch",type(x_batch),np.shape(x_batch)) 
            x_rs_batch = self.random_start(x = x_batch,clip_min = clip_min,clip_max = clip_max,eps = 0.01)
            
            

            # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)
            x_rs_batch_tanh = original_to_tanh(x_rs_batch, clip_min, clip_max, self._tanh_smoother)
            #print("x_batch",type(x_batch))
            #print("x_rs_batch",x_rs_batch.shape)
            # Initialize binary search:
            c_current = self.initial_const * np.ones(x_batch.shape[0])
            c_lower_bound = np.zeros(x_batch.shape[0])
            c_double = np.ones(x_batch.shape[0]) > 0

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = 1 * np.ones(x_batch.shape[0])
            best_x_adv_batch = x_rs_batch.copy()

            for bss in range(self.binary_search_steps):
                logger.debug(
                    "Binary search step %i out of %i (c_mean==%f)",
                    bss,
                    self.binary_search_steps,
                    np.mean(c_current),
                )
                nb_active = int(np.sum(c_current < self._c_upper_bound))
                logger.debug(
                    "Number of samples with c_current < _c_upper_bound: %i out of %i",
                    nb_active,
                    x_batch.shape[0],
                )
                if nb_active == 0:  # pragma: no cover
                    break
                learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

                # Initialize perturbation in tanh space:
                #x_adv_batch = x_batch.copy()
                #x_adv_batch_tanh = x_batch_tanh.copy()
                x_adv_batch = x_rs_batch.copy()
                x_adv_batch_tanh = x_rs_batch_tanh.copy()
                
                
                z_logits, l2dist, loss = self._loss(x_batch, x_adv_batch, y_batch, c_current)
                attack_success = loss - l2dist <= 0
                overall_attack_success = attack_success
                #print("bss_start",bss,l2dist)
                #improved_adv = attack_success & (l2dist < 1)#best_l2dist)

                
                for i_iter in range(self.max_iter):
                    logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                    logger.debug("Average Loss: %f", np.mean(loss))
                    logger.debug("Average L2Dist: %f", np.mean(l2dist))
                    logger.debug("Average Margin Loss: %f", np.mean(loss - l2dist))
                    logger.debug(
                        "Current number of succeeded attacks: %i out of %i",
                        int(np.sum(attack_success)),
                        len(attack_success),
                    )
                    #print("iter_start",i_iter,l2dist)

                    improved_adv = attack_success & (l2dist < best_l2dist)
                    logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))
                    
                    if np.sum(improved_adv) > 0:
                        best_l2dist[improved_adv] = l2dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                        #print("iter1",i_iter)
                        #break
                    
                    active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                    nb_active = int(np.sum(active))
                    logger.debug(
                        "Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i",
                        nb_active,
                        x_batch.shape[0],
                    )
                    if nb_active == 0:  # pragma: no cover
                        break

                    # compute gradient:
                    logger.debug("Compute loss gradient")
                    perturbation_tanh = -self._loss_gradient(
                        z_logits[active],
                        y_batch[active],
                        x_batch[active],
                        x_adv_batch[active],
                        x_adv_batch_tanh[active],
                        c_current[active],
                        clip_min,
                        clip_max,
                    )

                    # perform line search to optimize perturbation
                    # first, halve the learning rate until perturbation actually decreases the loss:
                    prev_loss = loss.copy()
                    best_loss = loss.copy()
                    best_lr = np.zeros(x_batch.shape[0])
                    halving = np.zeros(x_batch.shape[0])

                    for i_halve in range(self.max_halving):
                        #print("halving_start",i_halve,l2dist)
                        logger.debug(
                            "Perform halving iteration %i out of %i",
                            i_halve,
                            self.max_halving,
                        )
                        do_halving = loss[active] >= prev_loss[active]
                        logger.debug(
                            "Halving to be performed on %i samples",
                            int(np.sum(do_halving)),
                        )
                        if np.sum(do_halving) == 0:
                            #print("halving_end, break",i_halve,l2dist)
                            break
                        ###
                        #improved_adv = attack_success & (l2dist < 1)#best_l2dist)
                        #if np.sum(improved_adv) > 0:
                            #best_l2dist[improved_adv] = l2dist[improved_adv]
                            #best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                            #break
                        ###                        
                        active_and_do_halving = active.copy()
                        active_and_do_halving[active] = do_halving

                        lr_mult = learning_rate[active_and_do_halving]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                        new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                            x_batch[active_and_do_halving],
                            new_x_adv_batch,
                            y_batch[active_and_do_halving],
                            c_current[active_and_do_halving],
                        )

                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                        
                        ###
                        #print("halving_end",i_halve,l2dist)
                        attack_success = loss - l2dist <= 0
                        #overall_attack_success = attack_success
                        improved_adv = attack_success & (l2dist < best_l2dist)
                        if np.sum(improved_adv) > 0:
                            #best_l2dist[improved_adv] = l2dist[improved_adv]
                            best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                            #print("halving",i_halve)
                            break
                        ###
                        
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        learning_rate[active_and_do_halving] /= 2
                        halving[active_and_do_halving] += 1
                    learning_rate[active] *= 2

                    # if no halving was actually required, double the learning rate as long as this
                    # decreases the loss:
                    for i_double in range(self.max_doubling):
                        #print("doubling_start",i_double,l2dist)
                        logger.debug(
                            "Perform doubling iteration %i out of %i",
                            i_double,
                            self.max_doubling,
                        )
                        do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                        logger.debug(
                            "Doubling to be performed on %i samples",
                            int(np.sum(do_doubling)),
                        )
                        if np.sum(do_doubling) == 0:
                            break
                          
                        active_and_do_doubling = active.copy()
                        active_and_do_doubling[active] = do_doubling
                        learning_rate[active_and_do_doubling] *= 2

                        lr_mult = learning_rate[active_and_do_doubling]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                        new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                            x_batch[active_and_do_doubling],
                            new_x_adv_batch,
                            y_batch[active_and_do_doubling],
                            c_current[active_and_do_doubling],
                        )
                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                        
                        ###
                        #print("doubling_end",i_double,l2dist)
                        attack_success = loss - l2dist <= 0
                        #overall_attack_success = attack_success
                        improved_adv = attack_success & (l2dist < best_l2dist)
                        if np.sum(improved_adv) > 0:
                            #best_l2dist[improved_adv] = l2dist[improved_adv]
                            best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                            #print("doubling",i_double)
                            break
                        ###
                        
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        
                    learning_rate[halving == 1] /= 2

                    update_adv = best_lr[active] > 0
                    logger.debug(
                        "Number of adversarial samples to be finally updated: %i",
                        int(np.sum(update_adv)),
                    )

                    if np.sum(update_adv) > 0:
                        active_and_update_adv = active.copy()
                        active_and_update_adv[active] = update_adv
                        best_lr_mult = best_lr[active_and_update_adv]
                        for _ in range(len(x.shape) - 1):
                            best_lr_mult = best_lr_mult[:, np.newaxis]

                        x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                        best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                        x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1

                        x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                        x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv6, clip_min, clip_max)
                        (
                            z_logits[active_and_update_adv],
                            l2dist[active_and_update_adv],
                            loss[active_and_update_adv],
                        ) = self._loss(
                            x_batch[active_and_update_adv],
                            x_adv_batch[active_and_update_adv],
                            y_batch[active_and_update_adv],
                            c_current[active_and_update_adv],
                        )
                        attack_success = loss - l2dist <= 0
                        overall_attack_success = overall_attack_success | attack_success
                        
                        ###
                        #print("iter_end",i_iter,l2dist)
                        attack_success = loss - l2dist <= 0
                        improved_adv = attack_success & (l2dist < best_l2dist)
                        if np.sum(improved_adv) > 0:
                            #best_l2dist[improved_adv] = l2dist[improved_adv]
                            best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                            #print("iter",i_iter)
                            break
                        ###
                # Update depending on attack success:
                improved_adv = attack_success & (l2dist <best_l2dist)
                ###
                #print("bss_end",bss,l2dist)
                if np.sum(improved_adv) > 0:
                    #best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                    #print("bss",bss)
                    break
                ###
                logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))

                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                c_double[overall_attack_success] = False
                c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2

                c_old = c_current
                c_current[~overall_attack_success & c_double] *= 2

                c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]
            #print("l2",l2dist)
            x_adv[batch_index_1:batch_index_2] = best_x_adv_batch

        logger.info(
            "Success rate of C&W L_2 attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _check_params(self) -> None:
        if (
            not isinstance(
                self.binary_search_steps,
                int,
            )
            or self.binary_search_steps < 0
        ):
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_halving, int) or self.max_halving < 1:
            raise ValueError("The number of halving steps must be an integer greater than zero.")

        if not isinstance(self.max_doubling, int) or self.max_doubling < 1:
            raise ValueError("The number of doubling steps must be an integer greater than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")
