
# coding: utf-8

# # Using Dropout
# Let's see how we can use dropout for early stopping

from concept_dependency_graph import ConceptDependencyGraph
import data_generator as dg
from student import *
import simple_mdp as sm

import dynamics_model_class as dmc
import numpy as np
import dataset_utils
import tensorflow as tf
import tflearn
import copy
import time

def main():
    n_concepts = 4
    use_student2 = True
    student2_str = '2' if use_student2 else ''
    learn_prob = 0.5
    lp_str = '-lp{}'.format(int(learn_prob*100)) if not use_student2 else ''
    n_students = 100000
    seqlen = 7
    filter_mastery = False
    filter_str = '' if not filter_mastery else '-filtered'
    policy = 'random'
    filename = 'test{}-n{}-l{}{}-{}{}.pickle'.format(student2_str, n_students, seqlen,
                                                        lp_str, policy, filter_str)
    #concept_tree = sm.create_custom_dependency()
    concept_tree = ConceptDependencyGraph()
    concept_tree.init_default_tree(n_concepts)
    if not use_student2:
        test_student = Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    else:
        test_student = Student2(n_concepts)
    print(filename)



    # load toy data
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, filename))
    print('Average posttest: {}'.format(sm.expected_reward(data)))
    print('Percent of full posttest score: {}'.format(sm.percent_complete(data)))
    print('Percent of all seen: {}'.format(sm.percent_all_seen(data)))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)

    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    print(input_data_.shape)
    print(output_mask_.shape)
    print(target_data_.shape)

    # test_model hidden=16
    # test_model_mid hidden=10
    # test_model_small hidden=5
    # test_model_tiny hidden=3
    model_id = "test2_model_small"
    dropouts = np.array([1.0, 0.9, 0.8, 0.7])
    n_dropouts = dropouts.shape[0]
    total_epochs = 20
    reps = 20

    class ExtractCallback(tflearn.callbacks.Callback):
        def __init__(self):
            self.tstates = []
        def on_epoch_end(self, training_state):
            self.tstates.append(copy.copy(training_state))

    def test_dropout_losses():
        losses = np.zeros((n_dropouts,reps,total_epochs))
        val_losses = np.zeros((n_dropouts, reps,total_epochs))

        for d in range(n_dropouts):
            dropout = dropouts[d]
            for r in range(reps):
                ecall = ExtractCallback()
                dmodel = dmc.DynamicsModel(model_id=model_id, timesteps=seqlen, dropout=dropout, load_checkpoint=False)
                dmodel.train(train_data, n_epoch=total_epochs, callbacks=ecall, shuffle=False, load_checkpoint=False)
                losses[d,r,:] = np.array([s.global_loss for s in ecall.tstates])
                val_losses[d,r,:] = np.array([s.val_loss for s in ecall.tstates])

        return losses, val_losses

    losses, val_losses = test_dropout_losses()

    np.savez("dropoutput",dropouts=dropouts, losses=losses, vals=val_losses)

if __name__ == '__main__':
    starttime = time.time()

    np.random.seed()
    
    main()

    endtime = time.time()
    print('Time elapsed {}s'.format(endtime-starttime))




