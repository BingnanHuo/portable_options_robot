import argparse
import os
import random
import torch

#from evaluators import DivDisEvaluatorClassifier
from evaluation.evaluators.divdis_evaluator_classifier import DivDisEvaluatorClassifier
#from experiments.divdis_minigrid.core.advanced_minigrid_factored_divdis_classifier_experiment import \
#    AdvancedMinigridFactoredDivDisClassifierExperiment
from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.utils.utils import load_gin_configs, set_seed


img_dir = "resources/monte_images/"
# train using room 1 only
positive_train_files = [img_dir+"climb_down_ladder_room1_termination_positive.npy",
                        img_dir+"climb_down_ladder_room6_termination_positive.npy",]

negative_train_files = [img_dir+"climb_down_ladder_room1_termination_negative.npy",
                        img_dir+"screen_death_1.npy",
                        img_dir+"screen_death_2.npy",
                        img_dir+"screen_death_3.npy",
                        img_dir+"screen_death_4.npy",
                        img_dir+"climb_down_ladder_room6_termination_negative.npy",
                        img_dir+"climb_down_ladder_room6_uncertain.npy"
                        ]
initial_unlabelled_train_files = [
                            #img_dir+"screen_climb_down_ladder_initiation_positive.npy",
                            #img_dir+"screen_climb_down_ladder_initiation_negative.npy",
                            #img_dir+"climb_down_ladder_room0_initiation_positive.npy",
                            #img_dir+"climb_down_ladder_room0_initiation_negative.npy",
        ]
room_list = [0, 4, 3, 9, 8, 10, 11, 5, #12 here, has nothing
             13, 7, 6, 2, 14, 22, # 23 
             21, 19, 18]

initial_unlabelled_train_files = [
    # 0
    img_dir + "climb_up_ladder_room0_termination_positive.npy",
     img_dir + "climb_up_ladder_room0_termination_negative.npy",
     img_dir + "climb_up_ladder_room0_uncertain.npy",
     img_dir+"climb_down_ladder_room0_termination_negative.npy",
     img_dir+"climb_down_ladder_room0_uncertain.npy",
    # 4
    img_dir + "climb_up_ladder_room4_termination_negative.npy",
     img_dir + "climb_up_ladder_room4_uncertain.npy",
     img_dir+"climb_down_ladder_room4_termination_negative.npy",
     img_dir+"climb_down_ladder_room4_uncertain.npy",
     img_dir + "move_right_enemy_room4_termination_positive.npy",
     img_dir + "move_right_enemy_room4_termination_negative.npy",
     img_dir + "move_left_enemy_room4_termination_positive.npy",
     img_dir + "move_left_enemy_room4_termination_negative.npy",
    # 3
    img_dir + "climb_up_ladder_room3_termination_positive.npy",
     img_dir + "climb_up_ladder_room3_termination_negative.npy",
     img_dir + "climb_up_ladder_room3_uncertain.npy",
     img_dir+"climb_down_ladder_room3_termination_negative.npy",
     img_dir+"climb_down_ladder_room3_uncertain.npy",
     img_dir + "move_right_enemy_room3_termination_positive.npy",
     img_dir + "move_right_enemy_room3_termination_negative.npy",
     img_dir + "move_left_enemy_room3_termination_positive.npy",
     img_dir + "move_left_enemy_room3_termination_negative.npy",
    # 9
    img_dir + "climb_up_ladder_room9_termination_negative.npy",
     img_dir + "climb_up_ladder_room9_uncertain.npy",
     img_dir+"climb_down_ladder_room9_termination_positive.npy",
     img_dir+"climb_down_ladder_room9_termination_negative.npy",
     img_dir+"climb_down_ladder_room9_uncertain.npy",
     img_dir + "move_right_enemy_room9left_termination_positive.npy",
     img_dir + "move_right_enemy_room9left_termination_negative.npy",
     img_dir + "move_right_enemy_room9right_termination_positive.npy",
     img_dir + "move_right_enemy_room9right_termination_negative.npy",
     img_dir + "move_left_enemy_room9left_termination_positive.npy",
     img_dir + "move_left_enemy_room9left_termination_negative.npy",
     img_dir + "move_left_enemy_room9right_termination_positive.npy",
     img_dir + "move_left_enemy_room9right_termination_negative.npy",
    # 8
    img_dir + "room8_walk_around.npy",
    # 10
    img_dir + "climb_up_ladder_room10_termination_negative.npy",
     img_dir + "climb_up_ladder_room10_uncertain.npy",
     img_dir+"climb_down_ladder_room10_termination_negative.npy",
     img_dir+"climb_down_ladder_room10_termination_positive.npy",
     img_dir+"climb_down_ladder_room10_uncertain.npy",
    # 11
    img_dir + "climb_up_ladder_room11_termination_negative.npy",
     img_dir + "climb_up_ladder_room11_uncertain.npy",
     img_dir+"climb_down_ladder_room11_termination_negative.npy",
     img_dir+"climb_down_ladder_room11_uncertain.npy",
     img_dir + "move_right_enemy_room11left_termination_positive.npy",
     img_dir + "move_right_enemy_room11left_termination_negative.npy",
     img_dir + "move_right_enemy_room11right_termination_positive.npy",
     img_dir + "move_right_enemy_room11right_termination_negative.npy",
     img_dir + "move_left_enemy_room11left_termination_positive.npy",
     img_dir + "move_left_enemy_room11left_termination_negative.npy",
     img_dir + "move_left_enemy_room11right_termination_positive.npy",
     img_dir + "move_left_enemy_room11right_termination_negative.npy",
    # 5
    img_dir + "climb_up_ladder_room5_termination_positive.npy",
     img_dir + "climb_up_ladder_room5_termination_negative.npy",
     img_dir + "climb_up_ladder_room5_uncertain.npy",
     img_dir+"climb_down_ladder_room5_termination_negative.npy",
     img_dir+"climb_down_ladder_room5_uncertain.npy",
     img_dir + "move_right_enemy_room5_termination_positive.npy",
     img_dir + "move_right_enemy_room5_termination_negative.npy",
     img_dir + "move_left_enemy_room5_termination_positive.npy",
     img_dir + "move_left_enemy_room5_termination_negative.npy",
    # 13
    img_dir + "climb_up_ladder_room13_termination_negative.npy",
     img_dir + "climb_up_ladder_room13_uncertain.npy",
     img_dir+"climb_down_ladder_room13_termination_negative.npy",
     img_dir+"climb_down_ladder_room13_uncertain.npy",
     img_dir + "move_right_enemy_room13_termination_positive.npy",
     img_dir + "move_right_enemy_room13_termination_negative.npy",
     img_dir + "move_left_enemy_room13_termination_positive.npy",
     img_dir + "move_left_enemy_room13_termination_negative.npy",
    # 7
    img_dir + "climb_up_ladder_room7_termination_positive.npy",
     img_dir + "climb_up_ladder_room7_termination_negative.npy",
     img_dir + "climb_up_ladder_room7_uncertain.npy",
     img_dir+"climb_down_ladder_room7_termination_negative.npy",
     img_dir+"climb_down_ladder_room7_uncertain.npy",
    # 6
    img_dir + "climb_up_ladder_room6_termination_negative.npy",
     img_dir + "climb_up_ladder_room6_uncertain.npy",
     #img_dir+"climb_down_ladder_room6_termination_positive.npy",
     #img_dir+"climb_down_ladder_room6_termination_negative.npy",
     #img_dir+"climb_down_ladder_room6_uncertain.npy",
    # 2
    img_dir + "climb_up_ladder_room2_termination_positive.npy",
     img_dir + "climb_up_ladder_room2_termination_negative.npy",
     img_dir + "climb_up_ladder_room2_uncertain.npy",
     img_dir+"climb_down_ladder_room2_termination_negative.npy",
     img_dir+"climb_down_ladder_room2_uncertain.npy",
     img_dir + "move_right_enemy_room2_termination_positive.npy",
     img_dir + "move_right_enemy_room2_termination_negative.npy",
     img_dir + "move_left_enemy_room2_termination_positive.npy",
     img_dir + "move_left_enemy_room2_termination_negative.npy",
    # 14
    img_dir + "climb_up_ladder_room14_termination_positive.npy",
     img_dir + "climb_up_ladder_room14_termination_negative.npy",
     img_dir + "climb_up_ladder_room14_uncertain.npy",
     img_dir+"climb_down_ladder_room14_termination_negative.npy",
     img_dir+"climb_down_ladder_room14_uncertain.npy",
    # 22
    img_dir + "climb_up_ladder_room22_termination_negative.npy",
     img_dir + "climb_up_ladder_room22_uncertain.npy",
     img_dir+"climb_down_ladder_room22_termination_negative.npy",
     img_dir+"climb_down_ladder_room22_termination_positive.npy",
     img_dir+"climb_down_ladder_room22_uncertain.npy",
     img_dir + "move_right_enemy_room22_termination_positive.npy",
     img_dir + "move_right_enemy_room22_termination_negative.npy",
     img_dir + "move_left_enemy_room22_termination_positive.npy",
     img_dir + "move_left_enemy_room22_termination_negative.npy",
    # 21
    img_dir + "climb_up_ladder_room21_termination_negative.npy",
     img_dir + "climb_up_ladder_room21_uncertain.npy",
     img_dir+"climb_down_ladder_room21_termination_positive.npy",
     img_dir+"climb_down_ladder_room21_termination_negative.npy",
     img_dir+"climb_down_ladder_room21_uncertain.npy",
     img_dir + "move_right_enemy_room21_termination_positive.npy",
     img_dir + "move_right_enemy_room21_termination_negative.npy",
     img_dir + "move_left_enemy_room21_termination_positive.npy",
     img_dir + "move_left_enemy_room21_termination_negative.npy",
    # 19
    img_dir + "climb_up_ladder_room19_termination_negative.npy",
     img_dir + "climb_up_ladder_room19_uncertain.npy",
     img_dir+"climb_down_ladder_room19_termination_positive.npy",
     img_dir+"climb_down_ladder_room19_termination_negative.npy",
     img_dir+"climb_down_ladder_room19_uncertain.npy",
    # 18
    img_dir + "room18_walk_around.npy",
     img_dir + "move_left_enemy_room18_termination_positive.npy",
     img_dir + "move_left_enemy_room18_termination_negative.npy"
]


positive_test_files = [#img_dir+"climb_down_ladder_room6_termination_positive.npy",
                       img_dir+"climb_down_ladder_room9_termination_positive.npy",
                       img_dir+"climb_down_ladder_room10_termination_positive.npy",
                       img_dir+"climb_down_ladder_room19_termination_positive.npy",
                       img_dir+"climb_down_ladder_room21_termination_positive.npy",
                       img_dir+"climb_down_ladder_room22_termination_positive.npy",
                       ]
negative_test_files = [img_dir+"climb_down_ladder_room0_termination_negative.npy",
                       img_dir+"climb_down_ladder_room2_termination_negative.npy",
                       img_dir+"climb_down_ladder_room3_termination_negative.npy",
                       img_dir+"climb_down_ladder_room4_termination_negative.npy",
                       img_dir+"climb_down_ladder_room5_termination_negative.npy",
                       #img_dir+"climb_down_ladder_room6_termination_negative.npy",
                       img_dir+"climb_down_ladder_room7_termination_negative.npy",
                       img_dir+"climb_down_ladder_room9_termination_negative.npy",
                       img_dir+"climb_down_ladder_room10_termination_negative.npy",
                       img_dir+"climb_down_ladder_room11_termination_negative.npy",
                       img_dir+"climb_down_ladder_room13_termination_negative.npy",
                       img_dir+"climb_down_ladder_room14_termination_negative.npy",
                       img_dir+"climb_down_ladder_room19_termination_negative.npy",
                       img_dir+"climb_down_ladder_room21_termination_negative.npy",
                       img_dir+"climb_down_ladder_room22_termination_negative.npy",                       
                       ]
uncertain_test_files = [img_dir+"climb_down_ladder_room0_uncertain.npy",
                        img_dir+"climb_down_ladder_room2_uncertain.npy",
                        img_dir+"climb_down_ladder_room3_uncertain.npy",
                        img_dir+"climb_down_ladder_room4_uncertain.npy",
                        img_dir+"climb_down_ladder_room5_uncertain.npy",
                        #img_dir+"climb_down_ladder_room6_uncertain.npy",
                        img_dir+"climb_down_ladder_room7_uncertain.npy",
                        img_dir+"climb_down_ladder_room9_uncertain.npy",
                        img_dir+"climb_down_ladder_room10_uncertain.npy",
                        img_dir+"climb_down_ladder_room11_uncertain.npy",
                        img_dir+"climb_down_ladder_room13_uncertain.npy",
                        img_dir+"climb_down_ladder_room14_uncertain.npy",
                        img_dir+"climb_down_ladder_room19_uncertain.npy",
                        img_dir+"climb_down_ladder_room21_uncertain.npy",
                        img_dir+"climb_down_ladder_room22_uncertain.npy",
                        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--classifier_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
                ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
                ' "create_atari_environment.game_name="Pong"").')
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    set_seed(args.seed)

    classifier = DivDisClassifier(log_dir=args.base_dir+"logs")
    classifier.add_data(positive_train_files,
                        negative_train_files,
                        initial_unlabelled_train_files)
    classifier.train(200)

    

    evaluator = DivDisEvaluatorClassifier(
                    classifier,
                    base_dir=args.base_dir)
    evaluator.add_test_files(positive_test_files, negative_test_files)
    acc_pos, acc_neg, acc, weighted_acc = evaluator.test_classifier()
    print(f"weighted_acc: {weighted_acc}")
    print(f"raw acc: {acc}")
    print(f"acc_pos: {acc_pos}")
    print(f"acc_neg: {acc_neg}")

    evaluator.evaluate_images(25)

    #evaluator.add_true_from_files(positive_test_files)
    #evaluator.add_false_from_files(negative_test_files)
    #evaluator.evaluate(2)

    # print head complexity
    #print(evaluator.get_head_complexity())