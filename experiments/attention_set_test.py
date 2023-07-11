from portable.option.sets import TransformerSet
import torch

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_positive.npy',
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_negative.npy',
]

termination_positive_files = [
    'resources/minigrid_images/doorkey_getkey_0_termination_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_1_termination_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_2_termination_image_positive.npy',
]

termination_negative_files = [
    'resources/minigrid_images/doorkey_getkey_0_termination_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_1_termination_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_2_termination_image_negative.npy',
]

device = torch.device("cuda")
feature_size = 728
attention_set = 1
feed_forward_dim = 1025
lr = 1e-2
encoder_layer_num = 2

classifier = TransformerSet(device=device,
                            attention_num=attention_set,
                            feature_size=feature_size,
                            feature_extractor_type="factored_minigrid",
                            feature_num=6,
                            encoder_layer_num=encoder_layer_num,
                            feed_forward_hidden_dim=feed_forward_dim,
                            learning_rate=lr,
                            log_dir="runs/new_attention")

classifier.add_data_from_files(initiation_positive_files,
                               initiation_negative_files,
                               [])

classifier.train(10000)



