import os
import zipfile

_A1_FILES = [
    "pytorch101.py",
    "pytorch101.ipynb",
    "knn.py",
    "knn.ipynb",
]

_A2_FILES = [
    "linear_classifier.py",
    "linear_classifier.ipynb",
    "two_layer_net.py",
    "two_layer_net.ipynb",
    "challenge_problem.ipynb",
    "svm_best_model.pt",
    "softmax_best_model.pt",
    "nn_best_model.pt",
    "hand_drawn_weights.jpg",
]

_A3_FILES = [
    "fully_connected_networks.py",
    "fully_connected_networks.ipynb",
    "convolutional_networks.py",
    "convolutional_networks.ipynb",
    "best_overfit_five_layer_net.pth",
    "best_two_layer_net.pth",
    "one_minute_deepconvnet.pth",
    "overfit_deepconvnet.pth",
]

_A4_FILES = [
    "network_visualization.py",
    "network_visualization.ipynb",
    "style_transfer.py",
    "style_transfer.ipynb",
    "pytorch_autograd_and_nn.py",
    "pytorch_autograd_and_nn.ipynb",
    "rnn_lstm_attention_captioning.py",
    "rnn_lstm_attention_captioning.ipynb",
    # result files
    "pytorch_autograd_and_nn.pkl",
    "rnn_lstm_attention_submission.pkl",
    "saliency_maps_results.jpg",
    "adversarial_attacks_results.jpg",
    "class_viz_result.jpg",
    "style_transfer_result.jpg",
    "feature_inversion_result.jpg",
]

_A5_FILES = [
    "single_stage_detector.py",
    "two_stage_detector.py",
    "single_stage_detector_yolo.ipynb",
    "two_stage_detector_faster_rcnn.ipynb",
    "yolo_detector.pt",
    "frcnn_detector.pt",
]


_A6_FILES = [
    "vae.py",
    "gan.py",
    "variational_autoencoders.ipynb",
    "generative_adversarial_networks.ipynb",
    "vae_generation.jpg",
    "conditional_vae_generation.jpg",
    "fc_gan_results.jpg",
    "ls_gan_results.jpg",
    "dc_gan_results.jpg",
]


def make_a1_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A1_FILES, "P1", lastname, firstname)


def make_a2_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A2_FILES, "P2", lastname, firstname)


def make_a3_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A3_FILES, "P3", lastname, firstname)


def make_a4_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A4_FILES, "P4", lastname, firstname)


def make_a5_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A5_FILES, "P5", lastname, firstname)


def make_a6_submission(assignment_path, lastname=None, firstname=None):
    _make_submission(assignment_path, _A6_FILES, "P6", lastname, firstname)


def _make_submission(
    assignment_path, file_list, assignment_no, lastname=None, firstname=None
):
    if lastname is None or firstname is None:
        lastname, firstname = _get_user_info()
    zip_path = "{}_{}_{}.zip".format(lastname, firstname, assignment_no)
    zip_path = os.path.join(assignment_path, zip_path)
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename)


def _get_user_info():
    lastname = None
    firstname = None
    if lastname is None:
        lastname = input("Enter your last name (e.g. Smith): ")
    if firstname is None:
        firstname = input("Enter your first name (e.g. Susan): ")
    return lastname, firstname

